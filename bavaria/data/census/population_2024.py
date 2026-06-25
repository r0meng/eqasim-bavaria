import bavaria.data.census.population as population_stage
import os

"""
This stage always loads the 2024 base case census data,
regardless of the active scenario year. Used as the reference
baseline for computing per-Kreis license rates in projection scenarios.
"""

def configure(context):
    context.stage("bavaria.data.spatial.codes")
    context.config("data_path")
    context.config("bavaria.population_path", "bavaria/a1310c_202400.xla")

def execute(context):
    return population_stage._load_census_2024(context)

def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("bavaria.population_path"))):
        raise RuntimeError("Bavarian census data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("bavaria.population_path")))
