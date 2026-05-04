import csv
from app import commandsMap
#evaluating raw model on results
prefix_to_lang = {
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu"
}

def main():
    total = 0
    success = 0

    with open("benchmark_results_raw.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fileName = row["fileName"]
            expectedAction = row["expectedAction"].lower()
            rawTranscription = row["rawTranscription"].lower()
            
            if expectedAction == "none":
                continue
                
            lang_prefix = fileName.split("_")[0]
            total += 1
            
            lang_key = prefix_to_lang.get(lang_prefix)
            if lang_key and lang_key in commandsMap:
                valid_natives = [
                    native.lower() for native, action in commandsMap[lang_key].items()
                    if action.lower() == expectedAction
                ]
                # Checks if the exact native continuous string appears anywhere in the raw transcripion
                if any(native in rawTranscription for native in valid_natives):
                    success += 1

    accuracy = (success / total) * 100 if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()