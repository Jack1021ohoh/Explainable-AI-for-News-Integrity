"""
Claimify Extractor 詳細測試腳本

這個腳本會顯示：
1. 原始文章
2. 切分後的所有句子（含編號和評分）
3. 篩選結果（哪些被選中、哪些被過濾）
4. 每個句子經過三階段處理的詳細過程
5. 最終提取的 claims
"""

import os
import re
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

# 測試文章
TEST_ARTICLE = """
Tesla reported record quarterly revenue of $25.5 billion in Q3 2024, representing a 7% increase from the same period last year. CEO Elon Musk stated during the earnings call that "we expect to deliver 2 million vehicles this year."

The company's stock rose 12% following the announcement, making it the best single-day gain since January 2023. Analysts at Morgan Stanley upgraded their price target from $250 to $310, citing strong demand in China where Tesla sold 150,000 vehicles in September alone.

However, some investors remain skeptical about Tesla's ability to maintain growth amid increasing competition from BYD and other Chinese manufacturers. The electric vehicle market is expected to grow significantly in the coming years.

In summary, Tesla's Q3 2024 results exceeded expectations, but challenges lie ahead.
"""

TEST_QUESTION = "What are the key highlights from Tesla's Q3 2024 earnings?"


def print_separator(title: str, char: str = "=", width: int = 70):
    """印出分隔線"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def test_sentence_splitting():
    """測試 1: 顯示句子切分結果"""
    from extractor_claimify import ClaimifyExtractor
    
    print_separator("測試 1: 句子切分", "=")
    
    extractor = ClaimifyExtractor()
    sentences = extractor._split_into_sentences(TEST_ARTICLE)
    
    print(f"原始文章長度: {len(TEST_ARTICLE)} 字元")
    print(f"切分出 {len(sentences)} 個句子:\n")
    
    for i, sent in enumerate(sentences):
        # 顯示句子長度和詞數
        word_count = len(sent.split())
        print(f"[{i:2d}] ({word_count:2d} 詞) {sent[:80]}{'...' if len(sent) > 80 else ''}")
    
    return sentences


def test_prefilter_scoring(sentences: List[str]):
    """測試 2: 顯示篩選評分過程"""
    from extractor_claimify import ClaimifyExtractor
    
    print_separator("測試 2: 智慧篩選評分", "=")
    
    extractor = ClaimifyExtractor()
    
    print("評分規則:")
    print("  [跳過] 轉折/總結開頭、介紹句、推測語氣、意見、預測")
    print("  [+3分] 百分比、金額、大數字 (million/billion)")
    print("  [+2分] 年份、人名、引述詞、職位、變化數據、季度")
    print("  [+1分] 月份、適中長度(10-40詞)")
    print()
    
    results = []
    
    for idx, sent in enumerate(sentences):
        word_count = len(sent.split())
        score = 0
        reasons = []
        skip_reason = None
        
        # 檢查是否太短
        if word_count < 5:
            skip_reason = "太短 (< 5 詞)"
        else:
            # 檢查跳過模式
            for pattern in extractor.SKIP_PATTERNS:
                if re.search(pattern, sent, re.IGNORECASE):
                    skip_reason = f"符合跳過模式: {pattern[:30]}..."
                    break
            
            if not skip_reason:
                # 計算分數
                for pattern, weight in extractor.PRIORITY_PATTERNS:
                    if re.search(pattern, sent, re.IGNORECASE):
                        score += weight
                        # 找出匹配的內容
                        match = re.search(pattern, sent, re.IGNORECASE)
                        if match:
                            reasons.append(f"+{weight}: '{match.group()}'")
                
                # 長度加分
                if 10 <= word_count <= 40:
                    score += 1
                    reasons.append(f"+1: 適中長度({word_count}詞)")
        
        results.append({
            "idx": idx,
            "sentence": sent,
            "word_count": word_count,
            "score": score,
            "reasons": reasons,
            "skip_reason": skip_reason
        })
    
    # 顯示結果
    print("-" * 70)
    for r in results:
        status = "❌ 跳過" if r["skip_reason"] else f"✓ {r['score']:2d}分"
        print(f"[{r['idx']:2d}] {status}")
        print(f"     {r['sentence'][:60]}...")
        
        if r["skip_reason"]:
            print(f"     原因: {r['skip_reason']}")
        elif r["reasons"]:
            print(f"     加分: {', '.join(r['reasons'])}")
        print()
    
    # 顯示排序後的結果
    print_separator("篩選結果 (按分數排序)", "-")
    
    valid_results = [r for r in results if not r["skip_reason"]]
    valid_results.sort(key=lambda x: (-x["score"], x["idx"]))
    
    print(f"有效句子: {len(valid_results)} / {len(results)}")
    print(f"如果 max_sentences=5，會選擇以下句子:\n")
    
    for i, r in enumerate(valid_results[:5]):
        print(f"  #{i+1} [原idx={r['idx']}] {r['score']}分: {r['sentence'][:50]}...")
    
    return results


def test_full_extraction():
    """測試 3: 完整提取流程（含 LLM）"""
    from extractor_claimify import ClaimifyExtractor
    
    print_separator("測試 3: 完整提取流程 (使用 LLM)", "=")
    
    extractor = ClaimifyExtractor()
    
    print(f"問題: {TEST_QUESTION}")
    print(f"設定: max_sentences=5, use_prefilter=True, max_workers=1")
    print()
    print("開始處理...")
    print("-" * 70)
    
    # 先手動執行篩選，以便顯示更多細節
    all_sentences = extractor._split_into_sentences(TEST_ARTICLE)
    selected = extractor._prefilter_sentences(all_sentences, max_sentences=5)
    
    print(f"\n篩選結果: 從 {len(all_sentences)} 句中選出 {len(selected)} 句")
    print("選中的句子:")
    for orig_idx, sent in selected:
        print(f"  [idx={orig_idx}] {sent[:60]}...")
    print()
    
    # 執行完整提取
    print("=" * 70)
    print(" 開始三階段處理")
    print("=" * 70)
    
    for i, (orig_idx, sentence) in enumerate(selected):
        print(f"\n{'─' * 70}")
        print(f"📝 句子 {i+1}/{len(selected)} (原始 idx={orig_idx})")
        print(f"{'─' * 70}")
        print(f"原文: {sentence}")
        print()
        
        # 建立 excerpt
        excerpt = extractor._create_excerpt(
            all_sentences, orig_idx,
            extractor.max_preceding,
            extractor.max_following
        )
        print(f"上下文 (excerpt):")
        print(f"  {excerpt[:100]}...")
        print()
        
        # Stage 1: Selection
        print("🔍 Stage 1: Selection")
        contains_verifiable, modified = extractor._stage_selection(
            sentence, excerpt, TEST_QUESTION
        )
        
        if not contains_verifiable:
            print("   結果: ❌ 無可驗證內容")
            continue
        
        print(f"   結果: ✓ 包含可驗證內容")
        if modified and modified != sentence:
            print(f"   修改後: {modified}")
        else:
            print(f"   (未修改)")
        
        working_sentence = modified or sentence
        print()
        
        # Stage 2: Disambiguation
        print("🔗 Stage 2: Disambiguation")
        can_disambiguate, decontextualized = extractor._stage_disambiguation(
            working_sentence, excerpt, TEST_QUESTION
        )
        
        if not can_disambiguate:
            print("   結果: ❌ 無法消歧義")
            continue
        
        print(f"   結果: ✓ 消歧義成功")
        if decontextualized and decontextualized != working_sentence:
            print(f"   消歧義後: {decontextualized}")
        else:
            print(f"   (未修改)")
        
        final_sentence = decontextualized or working_sentence
        print()
        
        # Stage 3: Decomposition
        print("📋 Stage 3: Decomposition")
        claims = extractor._stage_decomposition(
            final_sentence, excerpt, TEST_QUESTION
        )
        
        if claims:
            print(f"   結果: ✓ 提取出 {len(claims)} 個 claims")
            for j, claim in enumerate(claims, 1):
                print(f"   [{j}] {claim}")
        else:
            print("   結果: ❌ 未提取出 claims")
    
    print()
    print_separator("處理完成", "=")


def test_comparison():
    """測試 4: 原文 vs Claims 對照表"""
    from extractor_claimify import ClaimifyExtractor
    
    print_separator("測試 4: 原文 vs Claims 對照表", "=")
    
    extractor = ClaimifyExtractor()
    
    result = extractor.extract(
        TEST_ARTICLE,
        question=TEST_QUESTION,
        max_sentences=5,
        use_prefilter=True,
        max_workers=1,
        verbose=False  # 關閉內建 verbose
    )
    
    print("統計:")
    print(f"  - 文章總句數: {result.sentences_total}")
    print(f"  - 被篩選掉: {result.sentences_filtered}")
    print(f"  - 實際處理: {result.sentences_processed}")
    print(f"  - 有 claims: {result.sentences_with_claims}")
    print(f"  - 無可驗證: {result.sentences_no_verifiable}")
    print(f"  - 無法消歧義: {result.sentences_ambiguous}")
    print()
    
    print("詳細對照:")
    print("-" * 70)
    
    for i, detail in enumerate(result.claim_details, 1):
        print(f"\n[Claim {i}]")
        print(f"  原句 (idx={detail['sentence_index']}): ")
        print(f"    {detail['source_sentence'][:70]}...")
        
        if detail['modified_sentence']:
            print(f"  Selection 後: ")
            print(f"    {detail['modified_sentence'][:70]}...")
        
        if detail['decontextualized_sentence']:
            print(f"  Disambiguation 後: ")
            print(f"    {detail['decontextualized_sentence'][:70]}...")
        
        print(f"  最終 Claim: ")
        print(f"    ➡️  {detail['claim']}")


def main():
    """主程式"""
    print("\n" + "🚀" * 35)
    print(" CLAIMIFY EXTRACTOR 詳細測試")
    print("🚀" * 35)
    
    print("\n📄 測試文章:")
    print("-" * 70)
    print(TEST_ARTICLE.strip())
    print("-" * 70)
    
    # 測試 1: 句子切分
    sentences = test_sentence_splitting()
    
    # 測試 2: 篩選評分（不需要 API）
    test_prefilter_scoring(sentences)
    
    # 詢問是否繼續（需要 API）
    print("\n" + "⚠️" * 35)
    print(" 以下測試需要呼叫 Groq API")
    print("⚠️" * 35)
    
    proceed = input("\n是否繼續? (y/n): ").strip().lower()
    
    if proceed == 'y':
        # 測試 3: 完整流程
        test_full_extraction()
        
        # 測試 4: 對照表
        test_comparison()
    else:
        print("\n已跳過 LLM 測試。")
    
    print("\n✅ 測試完成!")


if __name__ == "__main__":
    main()
