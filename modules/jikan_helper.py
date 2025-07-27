import requests

BASE_URL = "https://api.jikan.moe/v4"

def fetch_anime_info(title):
    try:
        response = requests.get(f"{BASE_URL}/anime", params={"q": title, "limit": 1})
        response.raise_for_status()
        data = response.json()
        if data["data"]:
            item = data["data"][0]
            return {
                "image_url": item["images"]["jpg"]["image_url"],
                "synopsis": item.get("synopsis", "No synopsis available."),
                "url": item.get("url", "")
            }
    except Exception as e:
        print(f"Jikan API error: {e}")
    return {
        "image_url": None,
        "synopsis": "No synopsis available.",
        "url": ""
    }
