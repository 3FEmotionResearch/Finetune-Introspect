#!/usr/bin/env python3
"""
Demonstration of improved emotion parsing on real messy outputs from your log.
"""
import re

def parse_emotions_old_way(raw_output):
    """Old parsing method - just basic splitting"""
    if raw_output:
        separators = [",", ";", " and "]
        emotion_words = [raw_output]
        
        for separator in separators:
            new_words = []
            for word in emotion_words:
                new_words.extend(word.split(separator))
            emotion_words = new_words
        
        emotions = []
        for word in emotion_words:
            clean_word = word.strip()
            if clean_word:
                emotions.append(clean_word)
        
        return emotions[:5]
    return ["Unknown"]

def parse_emotions_new_way(raw_output):
    """New improved parsing with multiple strategies"""
    if not raw_output:
        return ["Unknown"]
    
    # Strategy 1: Direct comma-separated parsing (ideal case)
    if ',' in raw_output and not any(char in raw_output for char in ['(', ')', '\n', 'Note', 'Translation']):
        emotions = []
        for word in raw_output.split(','):
            clean_word = re.sub(r'[^a-zA-Z]', '', word.strip()).strip()
            if clean_word and clean_word.isalpha() and 3 <= len(clean_word) <= 15:
                emotions.append(clean_word.capitalize())
        if emotions:
            return emotions[:5]
    
    # Strategy 2: Aggressive cleaning for messy outputs
    clean_output = raw_output.split('\n')[0]  # Take only first line
    
    # Remove unwanted patterns
    clean_output = re.sub(r'\([^)]*\)', '', clean_output)
    clean_output = re.sub(r'["\']', '', clean_output)
    clean_output = re.sub(r'\d+\.?\s*', '', clean_output)
    clean_output = re.sub(r'Note:.*', '', clean_output, flags=re.IGNORECASE)
    clean_output = re.sub(r'Translation:.*', '', clean_output, flags=re.IGNORECASE)
    clean_output = re.sub(r'Answer:.*?:', '', clean_output, flags=re.IGNORECASE)
    clean_output = re.sub(r'English.*?:', '', clean_output, flags=re.IGNORECASE)
    clean_output = re.sub(r'[_\-]{3,}', '', clean_output)
    clean_output = re.sub(r'etc\.?', '', clean_output, flags=re.IGNORECASE)
    clean_output = re.sub(r'Final Answer.*?:', '', clean_output, flags=re.IGNORECASE)
    
    # Strategy 3: Extract emotion words with validation
    potential_words = re.split(r'[,;\s\-\|\&\+]+', clean_output)
    
    valid_emotions = {
        'anger', 'angry', 'fear', 'fearful', 'afraid', 'sad', 'sadness', 'joy', 'happy',
        'frustration', 'frustrated', 'anxiety', 'anxious', 'worry', 'worried', 'helpless',
        'helplessness', 'desperation', 'desperate', 'rage', 'panic', 'guilt', 'shame'
    }
    
    emotions = []
    for word in potential_words:
        clean_word = re.sub(r'[^a-zA-Z]', '', word.strip()).lower()
        if (clean_word and clean_word.isalpha() and 3 <= len(clean_word) <= 15 and
            (clean_word in valid_emotions or len(clean_word) >= 4)):
            emotions.append(clean_word.capitalize())
    
    # Remove duplicates
    seen = set()
    unique_emotions = []
    for emotion in emotions:
        if emotion not in seen:
            seen.add(emotion)
            unique_emotions.append(emotion)
    
    return unique_emotions[:5] if unique_emotions else ["Unknown"]

def demo_improvements():
    """Demo the improvements using real messy outputs from your logs"""
    
    # Real messy outputs from your logs
    messy_outputs = [
        "(fear)  (anger)  (helplessness)\n\nNote: The text is in Simplified Chinese characters",
        "_______________________________________________________\n(Note: Please respond with English emotion words only",
        "1. 2. 3. 4. 5. \nFinal Answer: The final answer is: sad",
        "frustrated",  # This one is actually good
        "anxious, desperate, helpless\nTranslation: If you take me back there",
        "(fear)  (helplessness)  (anger)\nEnglish emotions: fear",
        "1. frustration\n2. fear\n3. anger\n4. desperation\n5. anxiety",
    ]
    
    print("🔍 DEMONSTRATION: Old vs New Emotion Parsing")
    print("=" * 60)
    
    for i, messy_output in enumerate(messy_outputs, 1):
        print(f"\n📝 Example {i}:")
        print(f"Raw Output: {messy_output[:80]}{'...' if len(messy_output) > 80 else ''}")
        print("-" * 40)
        
        old_result = parse_emotions_old_way(messy_output)
        new_result = parse_emotions_new_way(messy_output)
        
        print(f"❌ Old Method: {old_result}")
        print(f"✅ New Method: {new_result}")
        
        # Show improvement
        if len(new_result) > 0 and new_result[0] != "Unknown":
            if len(old_result) == 0 or old_result[0] == "Unknown" or len(new_result) > len([x for x in old_result if len(x) > 0 and x.isalpha()]):
                print("🎉 IMPROVEMENT: Much cleaner output!")
        print()

if __name__ == "__main__":
    demo_improvements() 