import json, requests

data = json.load(open("convo.json"))
resp = requests.post("http://localhost:8000/rate_v2", json=data)
out = resp.json()
# Write full response to file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Saved full response to output.json")