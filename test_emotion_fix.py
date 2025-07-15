#!/usr/bin/env python3
"""
Test script to validate the improved emotion inference functionality.
"""
import sys
import os
sys.path.append('.')
from infer_emotions import infer_emotion_labels, load_model

def test_emotion_inference():
    """Test the improved emotion inference on sample texts"""
    
    # Sample Chinese texts for testing
    test_texts = [
        "把我给抓回去那我就完了我肯定会受到责罚的",  # Current problematic text - should show fear/worry
        "我今天很开心，因为收到了好消息",  # Happy text
        "这件事让我感到很愤怒和失望",  # Angry/disappointed text
        "我很害怕会发生不好的事情",  # Fear text
        "看到这个场景我感到很悲伤"  # Sad text
    ]
    
    print("🧠 Testing Improved Emotion Inference")
    print("=" * 60)
    
    # Load model once
    print("Loading model...")
    load_model()
    print("Model loaded successfully!\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"📝 Test {i}: {text}")
        print("-" * 50)
        
        try:
            # Run inference
            emotions, raw_output = infer_emotion_labels(text)
            
            print(f"✅ Parsed Emotions: {emotions}")
            print(f"📄 Raw Output: {raw_output}")
            
            # Check if we got meaningful results
            if emotions and emotions != ["Unknown"]:
                print("🎉 SUCCESS: Got meaningful emotion words!")
            else:
                print("❌ ISSUE: Still getting empty or unknown emotions")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_emotion_inference() 