#!/usr/bin/env python3
"""
Test script to demonstrate improved emotion inference with consistent formatting.
"""
import sys
sys.path.append('.')
from infer_emotions import infer_emotion_labels

def test_emotion_inference():
    """Test the improved emotion inference on sample texts"""
    
    # Sample Chinese texts for testing
    test_texts = [
        "把我给抓回去那我就完了我肯定会受到责罚的",  # Current problematic text
        "我今天很开心，因为收到了好消息",  # Happy text
        "这件事让我感到很愤怒和失望",  # Angry/disappointed text
        "我很害怕会发生不好的事情",  # Fear text
        "看到这个场景我感到很悲伤"  # Sad text
    ]
    
    print("🧠 Testing Improved Emotion Inference")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 40)
        
        # Run inference 3 times to show consistency
        for run in range(1, 4):
            emotions, raw_output = infer_emotion_labels(text)
            print(f"Run {run}:")
            print(f"  ✅ Parsed Emotions: {emotions}")
            print(f"  📄 Raw Output: {raw_output[:100]}{'...' if len(raw_output) > 100 else ''}")
        
        print()

if __name__ == "__main__":
    test_emotion_inference() 