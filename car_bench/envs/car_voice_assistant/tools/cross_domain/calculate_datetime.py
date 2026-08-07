import json
from typing import Any, Dict

from car_bench.envs.tool import Tool


class CalculateDateTime(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], original_datetime: dict, times_to_add) -> str:
        # Add your code here
        for time in times_to_add:
            original_datetime["hour"] += time["hours"]
            original_datetime["minute"] += time["minutes"]
            if original_datetime["minute"] >= 60:
                original_datetime["hour"] += original_datetime["minute"] // 60
                original_datetime["minute"] = original_datetime["minute"] % 60
            if original_datetime["hour"] >= 24:
                original_datetime["day"] += original_datetime["hour"] // 24
                original_datetime["hour"] = original_datetime["hour"] % 24
            if original_datetime["day"] > 30:
                return "Error: calulation exceeds month - not supported"
        return json.dumps(original_datetime)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "calculate_datetime",
                "description": "Takes a datetime and adds specified times. It returns the new datetime",
                # "strict": True,
                "parameters": {
                    "type": "object",
                    "required": ["original_datetime", "times_to_add"],
                    "properties": {
                        "original_datetime": {
                            "type": "object",
                            "required": ["year", "month", "day", "hour", "minute"],
                            "properties": {
                                "year": {
                                    "type": "number",
                                    "description": "Year of the datetime",
                                },
                                "month": {
                                    "type": "number",
                                    "description": "Month of the datetime",
                                },
                                "day": {
                                    "type": "number",
                                    "description": "Day of the datetime",
                                },
                                "hour": {
                                    "type": "number",
                                    "description": "Hour of the datetime",
                                },
                                "minute": {
                                    "type": "number",
                                    "description": "Minute of the datetime",
                                },
                            },
                        },
                        "times_to_add": {
                            "type": "array",
                            "description": "Array of objects containing hours and minutes to add. Each object will be added to the original datetime",
                            "items": {
                                "type": "object",
                                "required": ["hours", "minutes"],
                                "properties": {
                                    "hours": {
                                        "type": "number",
                                        "description": "Number of hours to add",
                                    },
                                    "minutes": {
                                        "type": "number",
                                        "description": "Number of minutes to add",
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
