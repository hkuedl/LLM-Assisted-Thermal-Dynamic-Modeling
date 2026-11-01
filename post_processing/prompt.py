"""The original integrated prompt and its deterministic event/phase construction.
The V5 wording is intentional: packaging does not change model-visible input.
"""
from __future__ import annotations
import json
from copy import deepcopy
import numpy as np

SYSTEM_PROMPT = 'ROLE AND TASK\n\nEstimate a physically plausible event_gate for every supplied local\nthermostat-departure event. A deterministic operator already determines when\nthe correction acts and its direction; the gate only controls its magnitude.\nChoose a continuous event_gate between 0.00 and 1.00 inclusive, using at most\ntwo decimal places. Values between the interpretation anchors below are allowed.\n\nAfter completing that V5 event-level judgment, assign one phase_adjustment to\neach internal amplitude phase. The final phase gate is computed externally as\nclip(event_gate + phase_adjustment, 0, 1). Do not output the final phase gate.\n\nEVENT-LEVEL V5 PHYSICAL PRINCIPLES\n\n1. Evaluate each local event independently. Duration is the main persistence\n   cue: an event lasting roughly five hours or longer is persistent and\n   generally supports a stronger correction; a shorter event generally supports\n   a more conservative correction.\n2. Consider setpoint-shift magnitude and absolute departure dose together with\n   duration. Larger and more sustained departures support stronger correction.\n3. Direction determines whether the physical correction raises or lowers the\n   forecast, not how strong it should be.\n4. Equivalent local events with the same zone and building semantics should\n   receive comparable gates. Use semantic information only as a qualitative\n   modifier; it should not override clear duration and magnitude evidence.\n\nFor event_gate, use only v5_event_summary, building context, the frozen semantic\nprior and within-building semantic percentiles. Do not use internal phase\ntransitions to revise the V5 event-level judgment. Event and phase IDs are only\nidentifiers. Do not compare sibling events, OOD labels or building names as\nevidence.\n\nEVENT-GATE INTERPRETATION\n\n0.00 means no credible additional response, 0.25 weak, 0.50 moderate, 0.75\nstrong, and 1.00 nearly complete missing nominal response. Intermediate values\nare allowed.\n\nNo unseen temperature, target, residual, hindsight-optimal strength, previous\nmethod gate or performance metric may be inferred or requested.\n\nPHASE-LEVEL EXTENSION\n\n1. Start every phase_adjustment at 0. A phase adjustment is a local residual\n   relative to the already selected V5 event_gate, not a replacement event gate,\n   response magnitude, temperature or confidence score.\n2. Do not use phase amplitude alone to repeat the V5 event-level duration,\n   magnitude or dose judgment. Apply a nonzero adjustment only when a transition\n   mechanism supports a difference within the same continuous event.\n3. Transient same-direction amplification: when a contiguous phase becomes\n   substantially stronger in the same direction, the nominal correction can\n   increase before the incremental thermal response develops. If the stronger\n   phase is brief relative to the event and zone evidence supports slower\n   response, apply modest downward pressure. Sustained amplification or evidence\n   supporting faster response can justify zero adjustment.\n4. Carryover-aware same-direction attenuation: when a contiguous phase becomes\n   weaker while remaining in the same direction, the nominal correction can\n   shrink while response associated with the stronger preceding phase persists.\n   If continuity, preceding strength and zone inertia support carryover, apply\n   modest upward pressure. Conflicting evidence can justify zero adjustment.\n5. Direction reversals, returns to reference and mixed/non-comparable H/C\n   transitions do not inherit phase-level evidence across the boundary. Keep\n   unsupported adjustments at 0.\n6. Thermal inertia can delay response to amplification and preserve response\n   during attenuation, but it is not a universal upward or downward rule.\n   Other semantic fields are qualitative modifiers only.\n7. These are soft physical priors. Do not force phase ordering. Express weak or\n   conflicting evidence through confidence/reason and adjustments near zero.\n   No weighted-average or event-gate conservation constraint is imposed.\n\nFINAL CHECK\n\nConfirm that event_gate was selected by the original V5 event-level evidence and\nanchors before phase reasoning. Confirm that phase_adjustments encode only\nwithin-event transition effects and that no unobserved response or evaluation\nresult was used as fact.\n\nOUTPUT\n\nReturn JSON only with root key zone_gate_records, a list containing every\nsupplied zone exactly once. Each zone object contains zone and\nevent_gate_records. Return every requested event_id exactly once.\n\nEach event record contains only:\n- event_id: supplied integer\n- event_gate: JSON number in [0,1], at most two decimal places\n- confidence: low, medium or high\n- reason: concise V5 event-level duration/magnitude/dose and semantic basis\n- phase_adjustment_records: every requested phase_id exactly once\n\nEach phase-adjustment record contains only:\n- phase_id: supplied integer\n- phase_adjustment: JSON number in [-1,1], at most two decimal places\n- reason: concise transition mechanism, relevant zone modifier and limitation\n\nDo not add Markdown, final phase gates, previous-method comparisons, extra keys,\nzones, events or phases.'

EVENT_FIELDS = ("direction", "start_hour", "end_hour", "duration_hours",
    "mean_signed_center_shift_c", "mean_absolute_setpoint_shift_c",
    "peak_absolute_setpoint_shift_c", "absolute_departure_dose_c_hour")
SEMANTICS = ("Thermal_Inertia", "Occupancy_Volatility", "Env_Exposure",
    "Base_Physics_Trust", "Env_Sensitivity")
USER_PREFIX = (
    "For every supplied zone, judge each event independently. First select "
    "event_gate using only v5_event_summary and the supplied frozen semantics, "
    "following the original Continuous V5 event-level principles. Then inspect "
    "that event's internal phases and return phase_adjustments. Sibling events, "
    "event IDs, OOD labels and building names are not comparative evidence. "
    "Return all requested zones, events and phases in one JSON response."
)

def amplitude_phases(control_record):
    fields = [
        "reference_heating_profile_24h",
        "reference_cooling_profile_24h",
        "heating_profile_24h",
        "cooling_profile_24h",
    ]
    values = [control_record.get(field) for field in fields]
    if any(value is None for value in values):
        raise ValueError("Active episode is missing paired thermostat profiles")
    arrays = [np.asarray(value, dtype=float) for value in values]
    if (
        any(array.ndim != 1 for array in arrays)
        or len({len(array) for array in arrays}) != 1
        or not len(arrays[0])
        or not all(np.isfinite(array).all() for array in arrays)
    ):
        raise ValueError("Invalid thermostat profiles")
    heating_shift = arrays[2] - arrays[0]
    cooling_shift = arrays[3] - arrays[1]
    count = len(heating_shift)
    change = (np.abs(np.diff(heating_shift)) > 1e-8) | (
        np.abs(np.diff(cooling_shift)) > 1e-8
    )
    cuts = np.r_[0, np.flatnonzero(change) + 1, count]
    phases = []
    for start_index, end_index in zip(cuts[:-1], cuts[1:]):
        heating = float(heating_shift[start_index])
        cooling = float(cooling_shift[start_index])
        if max(abs(heating), abs(cooling)) <= 1e-8:
            continue
        center = (heating + cooling) / 2
        if center > 1e-8:
            direction = "upward"
        elif center < -1e-8:
            direction = "downward"
        else:
            direction = "deadband_or_symmetric"
        start = float(start_index * 24 / count)
        end = float(end_index * 24 / count)
        previous = phases[-1] if phases else None
        gap = start - previous["end_hour"] if previous else np.nan
        if previous is None:
            transition = "first_departure_today"
        elif gap > 1e-8:
            transition = "after_reference_interval"
        elif previous["direction"] == direction:
            transition = "same_direction_amplitude_change"
        else:
            transition = "direction_change"
        phases.append(
            {
                "start_hour": start,
                "end_hour": end,
                "direction": direction,
                "Heating_shift_C": round(heating, 8),
                "Cooling_shift_C": round(cooling, 8),
                "Center_shift_C": round(center, 8),
                "Transition_type": transition,
                "Previous_active_direction": previous["direction"] if previous else "none_today",
                "Previous_heating_shift_C": previous["Heating_shift_C"] if previous else np.nan,
                "Previous_cooling_shift_C": previous["Cooling_shift_C"] if previous else np.nan,
                "Gap_from_previous_active_hours": gap,
            }
        )
    return phases


def split_parent_episode(parent, phases):
    parent_start = float(parent["start_hour"])
    parent_end = float(parent["end_hour"])
    children = []
    for phase in phases:
        start = max(parent_start, phase["start_hour"])
        end = min(parent_end, phase["end_hour"])
        if end - start <= 1e-8:
            continue
        child = dict(
            parent,
            **{
                key: value
                for key, value in phase.items()
                if key not in ["start_hour", "end_hour"]
            }
        )
        child.update(
            start_hour=start,
            end_hour=end,
            Parent_episode=parent["episode_id"],
            Segment_within_parent=len(children) + 1,
            episode_id="P%s_S%d" % (parent["episode_id"], len(children) + 1),
        )
        children.append(child)
    covered = sum(child["end_hour"] - child["start_hour"] for child in children)
    if not children or not np.isclose(
        covered,
        parent_end - parent_start,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Split phases do not cover the full old episode")
    return children


def phase_relation(previous, current):
    if previous is None:
        return "new_or_maintained"
    before = [previous["Heating_shift_C"], previous["Cooling_shift_C"]]
    after = [current["Heating_shift_C"], current["Cooling_shift_C"]]
    if any(old * new < -1e-8 for old, new in zip(before, after)):
        return "other_or_unclear"
    differences = [abs(new) - abs(old) for old, new in zip(before, after)]
    if all(value >= -1e-8 for value in differences) and any(
        value > 1e-8 for value in differences
    ):
        return "same_direction_increase"
    if all(value <= 1e-8 for value in differences) and any(
        value < -1e-8 for value in differences
    ):
        return "same_direction_decrease"
    if all(abs(value) <= 1e-8 for value in differences):
        return "new_or_maintained"
    return "other_or_unclear"


def parse_and_validate(content, expected):
    text = content.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:]).rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    items = payload.get("zone_gate_records")
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("Zone coverage is missing or duplicated")
    if {item.get("zone") for item in items} != set(expected):
        raise ValueError("Zone set differs from request")
    result = {}
    for item in items:
        zone = item["zone"]
        events = item.get("event_gate_records", [])
        if len(events) != len(expected[zone]) or {
            event.get("event_id") for event in events
        } != set(expected[zone]):
            raise ValueError("Event coverage differs for " + zone)
        result[zone] = {}
        for event in events:
            event_id = event.get("event_id")
            if type(event_id) is not int or type(
                event.get("event_gate")
            ) not in [int, float]:
                raise ValueError("Event ID and gate must be numeric")
            gate = float(event["event_gate"])
            if not np.isfinite(gate) or not 0 <= gate <= 1:
                raise ValueError("Event gate is outside [0,1]")
            if (
                event.get("confidence") not in ["low", "medium", "high"]
                or not isinstance(event.get("reason"), str)
                or not event["reason"].strip()
            ):
                raise ValueError("Missing event confidence or reason")
            phases = event.get("phase_adjustment_records", [])
            expected_phases = expected[zone][event_id]
            if len(phases) != len(expected_phases) or {
                phase.get("phase_id") for phase in phases
            } != set(expected_phases):
                raise ValueError("Phase coverage differs for %s event %s" % (zone, event_id))
            phase_results = {}
            for phase in phases:
                phase_id = phase.get("phase_id")
                adjustment = phase.get("phase_adjustment")
                if type(phase_id) is not int or type(adjustment) not in [int, float]:
                    raise ValueError("Phase ID and adjustment must be numeric")
                adjustment = float(adjustment)
                if not np.isfinite(adjustment) or not -1 <= adjustment <= 1:
                    raise ValueError("Phase adjustment is outside [-1,1]")
                if not isinstance(phase.get("reason"), str) or not phase["reason"].strip():
                    raise ValueError("Missing phase reason")
                phase_results[phase_id] = {
                    "phase_adjustment": adjustment,
                    "reason": phase["reason"],
                }
            result[zone][event_id] = {
                "event_gate": gate,
                "confidence": event["confidence"],
                "reason": event["reason"],
                "phases": phase_results,
            }
    return result


def decode_phase_gates(event_gate, phases, phase_results):
    adjustments = np.asarray(
        [
            phase_results[int(phase["phase_id"])]["phase_adjustment"]
            for phase in phases
        ],
        dtype=float,
    )
    if not len(adjustments) or not np.isfinite(adjustments).all():
        raise ValueError("Invalid phase adjustments")
    unclipped = float(event_gate) + adjustments
    gates = np.clip(unclipped, 0.0, 1.0)
    return gates, unclipped



def records(payload, key):
    value = payload.get(key, {})
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        raise ValueError('Invalid '+key)
    names = [x.get('zone') for x in value]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate zones in '+key)
    return {x['zone']:x for x in value}


def zone_record(value, zone):
    if zone in value:
        return value[zone]
    token = lambda x: ''.join(c for c in str(x).lower() if c in 'abcdefghijklmnopqrstuvwxyz0123456789')
    matches = [r for z,r in value.items() if token(z)==token(zone)]
    if len(matches) != 1:
        raise ValueError('Missing/ambiguous control for '+zone)
    return matches[0]


def build_request(control, evidence_source, building, ood):
    """Accept existing neutral evidence or the original evidence container.

    Only the eight event fields and frozen evidence are selected; old gate,
    votes, targets and historical performance never enter a message.
    """
    neutral = 'zone_event_records' in evidence_source
    source = records(evidence_source, 'zone_event_records' if neutral else 'zone_gate_records')
    controls = records(control, 'control_records')
    cards, active, inactive = [], {}, {}
    for zone, record in source.items():
        parents = record.get('event_records' if neutral else 'episode_gate_records', [])
        if not parents:
            inactive[zone] = {'zone':zone, 'episode_gate_records':[]}
            continue
        evidence = record.get('evidence', {})
        semantics = evidence.get('frozen_semantic_prior', {})
        values = np.asarray(list(semantics.values()), float)
        if set(semantics)!=set(SEMANTICS) or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
            raise ValueError('Incomplete/invalid frozen semantics: '+zone)
        card = {'zone':zone,
                'building_context':evidence.get('building_context', {'building':building}),
                'frozen_semantic_prior':semantics,
                'within_building_semantic_percentile':evidence.get('within_building_semantic_percentile', {}),
                'requested_event_ids':[], 'event_context':[]}
        events = []
        all_phases = amplitude_phases(zone_record(controls,zone))
        for source_parent in parents:
            # Discard any saved gate values by constructing an evidence-only parent.
            parent = {key:source_parent[key] for key in EVENT_FIELDS}
            parent['episode_id'] = source_parent.get('episode_id',source_parent.get('event_id'))
            if parent['episode_id'] is None:
                raise ValueError('Parent event lacks identity for '+zone)
            children = split_parent_episode(parent,all_phases)
            event_id = len(events)+1
            descriptors, phase_cards, previous = [], [], None
            for phase_id,child in enumerate(children,1):
                descriptor = {
                    'episode_id':sum(len(x['phases']) for x in events)+phase_id,
                    'event_id':event_id,'phase_id':phase_id,'parent_episode_id':parent['episode_id'],
                    'start_hour':child['start_hour'],'end_hour':child['end_hour'],
                    'duration_hours':child['end_hour']-child['start_hour'],
                    'heating_departure_C':child['Heating_shift_C'],
                    'cooling_departure_C':child['Cooling_shift_C'],
                    'direction':child['direction'],'phase_relation':phase_relation(previous,child),
                    'event_elapsed_hours_before_phase':child['start_hour']-float(parent['start_hour'])}
                descriptors.append(descriptor)
                phase_cards.append({k:v for k,v in descriptor.items() if k not in ('parent_episode_id','episode_id')})
                previous=child
            card['requested_event_ids'].append(event_id)
            event_card = {'event_id':event_id,'ood_label':ood,
                'start_hour':parent['start_hour'],'end_hour':parent['end_hour'],
                'duration_hours':float(parent['end_hour'])-float(parent['start_hour']),
                'requested_phase_ids':[x['phase_id'] for x in descriptors],
                'phases':phase_cards,'temperature_response_observations':'not supplied'}
            summary={k:parent[k] for k in EVENT_FIELDS}
            summary['direction']=str(summary['direction'])
            for key in EVENT_FIELDS[1:]:summary[key]=float(summary[key])
            event_card['v5_event_summary']=summary
            card['event_context'].append(event_card)
            events.append({'event_id':event_id,'parent_episode_id':parent['episode_id'],'phases':descriptors})
        cards.append(card)
        active[zone]=events
    if not cards:
        raise ValueError('No active zones for %s/%s'%(building,ood))
    return {'building':building,'ood':ood,'cards':cards,'active':active,'inactive':inactive,
        'expected':{c['zone']:{e['event_id']:e['requested_phase_ids'] for e in c['event_context']} for c in cards}}


# Only method/version labels change. All physical principles and numerical anchors
# are inherited from the same original string, not rewritten independently.
PUBLIC_SYSTEM_PROMPT = (SYSTEM_PROMPT
    .replace('v5_event_summary', 'event_summary')
    .replace('original V5 ', '')
    .replace('V5 ', ''))
PUBLIC_USER_PREFIX = (USER_PREFIX
    .replace('v5_event_summary', 'event_summary')
    .replace('original Continuous V5 ', ''))


def messages(cards, style='original'):
    if style not in ('original', 'public'):
        raise ValueError('prompt_style must be original or public')
    if style == 'public':
        cards=deepcopy(cards)
        for card in cards:
            for event in card['event_context']:
                event['event_summary']=event.pop('v5_event_summary')
        system,prefix=PUBLIC_SYSTEM_PROMPT,PUBLIC_USER_PREFIX
    else:
        system,prefix=SYSTEM_PROMPT,USER_PREFIX
    return [{'role':'system','content':system}, {'role':'user','content':prefix+'\n\n'+json.dumps(cards,ensure_ascii=False,separators=(',',':'))}]


def batches(request,zones_per_call,style='original'):
    """Same event -> zone-batch protocol as the resumed original experiment."""
    if zones_per_call<1:raise ValueError('zones_per_call must be positive')
    result=[]
    for event_id in sorted({e['event_id'] for c in request['cards'] for e in c['event_context']}):
        local=[]
        for card in request['cards']:
            matching=[e for e in card['event_context'] if e['event_id']==event_id]
            if matching:
                local.append(dict(card,requested_event_ids=[event_id],event_context=matching))
        for start in range(0,len(local),zones_per_call):
            chunk=local[start:start+zones_per_call]
            result.append({'messages':messages(chunk,style), 'expected':{c['zone']:{event_id:request['expected'][c['zone']][event_id]} for c in chunk},
                'event_id':event_id,'batch':start//zones_per_call+1})
    return result


def aggregate(request, samples, metadata=None):
    """Mean event + mean adjustment, then clip; never mean of clipped votes."""
    if not samples:raise ValueError('No complete samples')
    final=dict(request['inactive'])
    n=len(samples)
    for zone,events in request['active'].items():
        episodes=[]
        for event in events:
            event_id=event['event_id']
            votes=[s[zone][event_id] for s in samples]
            gates=[v['event_gate'] for v in votes]
            mean_gate=float(np.mean(gates))
            counts={level:sum(v['confidence']==level for v in votes) for level in ('low','medium','high')}
            confidence=max(counts,key=counts.get)
            for phase in event['phases']:
                pv=[v['phases'][phase['phase_id']] for v in votes]
                adjustments=[x['phase_adjustment'] for x in pv]
                adjustment=float(np.mean(adjustments));unclipped=mean_gate+adjustment
                episodes.append(dict(phase, correction_gate=float(np.clip(unclipped,0,1)),
                    event_gate=mean_gate,phase_adjustment=adjustment,unclipped_phase_gate=unclipped,
                    was_clipped=bool(not np.isclose(np.clip(unclipped,0,1),unclipped)),
                    confidence=confidence,reason='mean of %d phase judgments'%n,
                    event_reason='mean of %d event judgments'%n,
                    sample_event_gates=gates,sample_adjustments=adjustments,
                    sample_confidences=[v['confidence'] for v in votes],
                    sample_event_reasons=[v['reason'] for v in votes],sample_phase_reasons=[v['reason'] for v in pv]))
        final[zone]={'zone':zone,'episode_gate_records':episodes}
    meta=dict(metadata or {},n_runs=n,aggregation='arithmetic_mean',
        phase_gate_decoder='clip(mean(event_gate) + mean(phase_adjustment), 0, 1)')
    return {'metadata':meta,'zone_gate_records':final}
