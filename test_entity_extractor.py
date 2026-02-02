import sys
import os

# 確保 Python 能找到 services 資料夾
sys.path.append(os.getcwd())

from services.entity_extractor import EntityExtractor

def main():
    # 定義多個測試案例
    test_cases = [
        {
            "desc": "案例 1: 標準混合 (John Doe)",
            "text": "John Doe 是一名軟體工程師，現居於台北。他在 Google 工作。"
        },
        {
            "desc": "案例 2: 三段式英文名 (William Henry Gates)",
            "text": "微軟的創辦人是 William Henry Gates，他對 AI 發展很有貢獻。"
        },
        {
            "desc": "案例 3: 科技名詞干擾 (Python/Java)",
            "text": "我們公司使用 Python 和 JavaScript 來開發，而不是用 C++。"
        },
        {
            "desc": "案例 4: 容易誤判的普通英文 (Hello World)",
            "text": "他在程式碼裡寫下了 Hello World，這不是一間公司，只是一句問候。"
        },
        {
            "desc": "案例 5: 純中文人名與複雜組織",
            "text": "王大明最近加入了 台灣積體電路製造股份有限公司 (TSMC) 的研發部門。"
        }
    ]

    print("🚀 開始批量測試...\n")

    try:
        extractor = EntityExtractor()
        
        for i, case in enumerate(test_cases, 1):
            print("=" * 60)
            print(f"[{case['desc']}]")
            print(f"📄 文本: {case['text']}")
            print("-" * 60)
            
            entities = extractor.extract_entities(case['text'])
            
            if not entities:
                print("   (無提取結果)")
            
            for entity in entities:
                label_map = {
                    "PERSON": "人物",
                    "ORG": "組織/公司",
                    "GPE": "地點/國家",
                    "LOC": "地理位置",
                    "DATE": "日期",
                    "WORK_OF_ART": "作品",
                    "PRODUCT": "產品"
                }
                zh_label = label_map.get(entity['label'], entity['label'])
                # 使用簡單的格式輸出
                print(f"   🔹 實體: {entity['text']:<15} | 類型: {entity['label']} ({zh_label})")
            print("\n")

    except Exception as e:
        print(f"\n❌ 錯誤: {e}")

if __name__ == "__main__":
    main()