"""Names shared by the runner and the archive comparison utility."""

BUILDINGS = ("118zone_Healthcare_1", "67zone_HotelSmall_1", "46zone_SchoolSecondary_1",
             "25zone_SchoolPrimary_1", "22zone_HotelLarge_B_1",
             "18zone_OfficeMedium_B_1", "10zone_RetailStripmall_C_1")
DEFAULT_BUILDINGS = ("67zone_HotelSmall_1", "46zone_SchoolSecondary_1", "18zone_OfficeMedium_B_1")
METHODS = {
    "data": "PI + Data Shape AutoCluster CNN",
    "llm": "PI + LLM Cluster CNN",
    "pi-film": "PI + Zone FiLM CNN",
    "data-film": "PI + Data Shape AutoCluster + Zone FiLM CNN",
    "llm-film": "PI + LLM Cluster + Zone FiLM CNN",
}
ARCHIVE_METHODS = {
    "pi-film": "PI CNN",
    "data-film": "PI + Data Shape AutoCluster CNN",
    "llm-film": "PI + LLM Cluster CNN",
}
