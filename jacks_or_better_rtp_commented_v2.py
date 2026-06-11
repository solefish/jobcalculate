"""
================================================================================
Jacks or Better 還元率（RTP）完全計算スクリプト
================================================================================

【このスクリプトが計算するもの】

  還元率（RTP: Return To Player）とは、
  「1コイン賭けたとき、長期的に平均で何コイン返ってくるか」を表す数値です。
  RTP 99.54% なら、100コイン賭けると平均99.54コイン戻ってくる計算になります。

  このスクリプトは「最適戦略を取り続けた場合のRTP」を計算します。
  つまり「どのカードを残すか」を常に最も期待値の高い選択をしたと仮定します。

【計算方法の概要】

  ドローポーカーは「5枚受け取る → 残したいカードを選ぶ → 残りを引き直す」
  という流れです。

  RTPを正確に求めるには、起こりうる全パターンを漏れなく計算する必要があります。

  Step 1: 最初に配られる5枚の手札、全C(52,5)=2,598,960通りを列挙する
  Step 2: 各手札について「どれを残すか」の全2^5=32パターンを評価する
  Step 3: 各パターンについて「残りデッキから引き直した後の期待値」を
          全組み合わせを列挙して正確に計算する
  Step 4: 32パターンのうち期待値が最大のものを「最適選択」とする
  Step 5: 全2,598,960手の最適期待値を平均したものがRTP

  この計算量は膨大なので、GPUで並列処理します。

【高速化の工夫：巨大ルックアップテーブル】

  「5枚の手札が何点か」を毎回計算するのではなく、
  事前に全パターンの答えを表（ルックアップテーブル）に書き込んでおきます。

  カード番号0〜51の5枚の並びを1つの整数インデックスに変換し、
  そのインデックスで表を引くだけで配当がわかります。

  インデックスの計算式:
    k = c0×52⁴ + c1×52³ + c2×52² + c3×52 + c4

  この表のサイズは 52^5 = 380,204,032エントリ（約726MB）。
  ソートも二分探索も不要で、1命令で配当を引けます。

================================================================================
"""

# --- ライブラリのインポート ---
import numpy as np               # 数値計算・配列操作
from numba import cuda           # GPU（CUDA）プログラミング
import math                      # 切り上げ除算などに使用
import time                      # 経過時間の計測
from itertools import combinations, permutations  # 組み合わせ・順列の列挙
from collections import Counter  # 出現回数の集計

# ==============================================================================
# 定数定義
# ==============================================================================

# デッキはカード番号0〜51で表現する
#
#   カード番号 c について:
#     スート（マーク）= c // 13   → 0=s, 1=h, 2=d, 3=c
#     ランク（数字）  = c % 13    → 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8,
#                                    7=9, 8=10, 9=J, 10=Q, 11=K, 12=A
#
#   例: カード番号13 → スート1(h), ランク0(2) → 2h
#       カード番号25 → スート1(h), ランク12(A) → Ah

DECK_SIZE  = 52       # デッキの枚数
TABLE_SIZE = 52 ** 5  # ルックアップテーブルのエントリ数 = 380,204,032

RANK_J = 9  # Jのランクインデックス（9=J, 10=Q, 11=K, 12=A）
            # ジャックスオアベターはランクが9以上のペアで成立する

# --- ストレート・ストレートフラッシュ判定用のビットマスク一覧 ---
#
# 各ランクを1ビットで表現し、5枚のランクの組み合わせをビット列で管理します。
# 例: [2,3,4,5,6] のストレートは ビット0〜4が立つ = 0b11111 = 31
#
# ロウ（A-2-3-4-5）ストレートは特殊で、AをビットAとしてもビット-1としても
# 扱えないため、A=ビット12 として A-2-3-4-5 を専用マスクで表現します。
#
# SF_WINDOWS_LIST[9] = 10-J-Q-K-A（ロイヤルフラッシュのランク構成）

SF_WINDOWS_LIST = [
    (1<<12)|(1<<0)|(1<<1)|(1<<2)|(1<<3),    # A-2-3-4-5（ロウストレート）
    (1<<0) |(1<<1)|(1<<2)|(1<<3)|(1<<4),    # 2-3-4-5-6
    (1<<1) |(1<<2)|(1<<3)|(1<<4)|(1<<5),    # 3-4-5-6-7
    (1<<2) |(1<<3)|(1<<4)|(1<<5)|(1<<6),    # 4-5-6-7-8
    (1<<3) |(1<<4)|(1<<5)|(1<<6)|(1<<7),    # 5-6-7-8-9
    (1<<4) |(1<<5)|(1<<6)|(1<<7)|(1<<8),    # 6-7-8-9-10
    (1<<5) |(1<<6)|(1<<7)|(1<<8)|(1<<9),    # 7-8-9-10-J
    (1<<6) |(1<<7)|(1<<8)|(1<<9)|(1<<10),   # 8-9-10-J-Q
    (1<<7) |(1<<8)|(1<<9)|(1<<10)|(1<<11),  # 9-10-J-Q-K
    (1<<8) |(1<<9)|(1<<10)|(1<<11)|(1<<12), # 10-J-Q-K-A（ロイヤルフラッシュ）
]
ROYAL_WINDOW = SF_WINDOWS_LIST[9]  # ロイヤルフラッシュのランクビットマスク

def card_name(c):
    """カード番号を人間が読める文字列に変換する（例: 25 → "Ah"）"""
    return "23456789TJQKA"[c % 13] + "shdc"[c // 13]

# ==============================================================================
# CPU版 役判定関数
# ==============================================================================

def classify_cpu(hand):
    """
    5枚の手札（カード番号のリスト）を受け取り、配当倍率を返す。

    この関数はCPUで実行される。主な用途はルックアップテーブルの事前構築。
    （GPU上での役判定はテーブル参照に置き換えることで高速化している）

    判定の優先順位（強い役から順に判定する）:
      ロイヤルフラッシュ → ストレートフラッシュ → フォーカード →
      フルハウス → フラッシュ → ストレート → スリーカード →
      ツーペア → ジャックスオアベター → ハズレ
    """
    # 各カードからランクとスートを取り出す
    ranks = [c % 13  for c in hand]   # 例: [8, 9, 10, 11, 12]（10-J-Q-K-A）
    suits = [c // 13 for c in hand]   # 例: [0, 0,  0,  0,  0]（全部s）

    # ランクの出現回数を数える（例: [J,J,K,K,A] → {J:2, K:2, A:1}）
    nrc = Counter(ranks)

    # 出現回数を多い順に並べる（例: [2, 2, 1]）
    counts = sorted(nrc.values(), reverse=True)

    # フラッシュ判定: 全カードのスートが同じか
    is_flush = (len(set(suits)) == 1)

    # ランクの種類集合（重複を除いたランク一覧）
    rank_set = set(ranks)

    # ストレート判定（内部関数）
    def is_straight():
        # ランクが5種類でなければストレートにならない（ペアがある = NG）
        if len(rank_set) != 5:
            return False
        # 5枚のランクをビット列に変換する
        # 例: [8,9,10,11,12] → ビット8〜12が立つ整数
        rs = 0
        for r in ranks:
            rs |= (1 << r)
        # SF_WINDOWS_LIST の各ストレート形と比較する
        for w in SF_WINDOWS_LIST:
            if rs == w:
                return True
        return False

    is_st = is_straight()

    # ランクのビット列を作成（ロイヤルフラッシュ判定にも使う）
    rs = 0
    for r in ranks:
        rs |= (1 << r)

    # --- 役判定（強い順） ---

    # ロイヤルフラッシュ: フラッシュ かつ 10-J-Q-K-A のランク構成
    if is_flush and rs == ROYAL_WINDOW:
        return 800

    # ストレートフラッシュ: フラッシュ かつ ストレート（ロイヤル以外）
    if is_flush and is_st:
        return 50

    # フォーカード: 同じランクが4枚
    if counts[0] == 4:
        return 25

    # フルハウス: 3枚 + 2枚の組み合わせ
    if counts[:2] == [3, 2]:
        return 9

    # フラッシュ: 全部同じスート（ストレートでない）
    if is_flush:
        return 6

    # ストレート: 連続するランク5枚（同スートでない）
    if is_st:
        return 4

    # スリーカード: 同じランクが3枚
    if counts[0] == 3:
        return 3

    # ツーペア: ペアが2組
    if counts[:2] == [2, 2]:
        return 2

    # ジャックスオアベター: J・Q・K・A のいずれかでペアが1組
    # （10以下のペアは配当なし）
    if counts[0] == 2:
        for rank, cnt in nrc.items():
            if cnt == 2 and rank >= RANK_J:  # RANK_J = 9（Jのランク）
                return 1

    # ハズレ: 上記のどれにも該当しない
    return 0

# ==============================================================================
# 巨大ルックアップテーブルの構築
# ==============================================================================

def build_giant_table(all_hands):
    """
    52^5 = 380,204,032エントリのルックアップテーブルを構築して返す。

    【テーブルの仕組み】

      カード5枚 (c0, c1, c2, c3, c4) を以下の式で1つの整数インデックスに変換する:
        k = c0×52⁴ + c1×52³ + c2×52² + c3×52 + c4

      これは「52進数の5桁の数」と考えればよい。
      例: カード[10, 23, 35, 48, 0] なら k = 10×52⁴ + 23×52³ + ...

      table[k] にその手札の配当が格納されているので、
      GPU側はこの式でkを計算してtable[k]を引くだけでよい。

    【なぜ全順列が必要か】

      C(52,5)=2,598,960通りはランク・スートが異なる組み合わせで、
      順序は考慮しない（例: [10,23,35,48,0] と [0,10,23,35,48] は同じ手）。

      しかしGPU側では「ホールドしたカード + 引いたカード」をそのままの順序で
      インデックスに変換するので、どんな順序でも正しい配当を返せなければならない。

      そこで、各手札の5!=120通りの全順列に対して同じ配当を書き込む。
      C(52,5) × 120 = 311,875,200エントリが有効なデータとなる。
      残り（重複カードを含む無効なインデックス）は0のまま（ハズレ）。

    【メモリ】
      int16（2バイト）× 380,204,032 ≈ 726MB
    """
    N = len(all_hands)
    print(f"  テーブルサイズ: {TABLE_SIZE:,} エントリ ({TABLE_SIZE*2/1024**3:.2f} GB)")

    # テーブルを0（ハズレ）で初期化する
    table = np.zeros(TABLE_SIZE, dtype=np.int16)

    print(f"  {N:,}通りの役判定 + 全順列展開中...")
    t0 = time.time()

    # 52の累乗を事前に計算しておく（インデックス計算を高速化）
    # P[0]=1, P[1]=52, P[2]=52², P[3]=52³, P[4]=52⁴
    P = [1, 52, 52**2, 52**3, 52**4]

    # 役別の手数を集計するカウンタ（統計表示用）
    pay_counts = Counter()

    for i, hand in enumerate(all_hands):
        # この手札の配当を判定する（CPUで実行）
        payout = classify_cpu(list(hand))
        pay_counts[payout] += 1  # 統計用に記録

        if payout > 0:
            # 配当がある手札のみ、全120通りの順列をテーブルに書き込む
            # （配当0=ハズレはテーブルの初期値0と一致するので書き込み不要）
            for perm in permutations(hand):
                idx = (perm[0]*P[4] + perm[1]*P[3] +
                       perm[2]*P[2] + perm[3]*P[1] + perm[4])
                table[idx] = payout

        # 進捗表示（50万手ごと）
        if (i + 1) % 500000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"    {i+1:,}/{N:,} ({elapsed:.1f}秒経過 残り約{eta:.0f}秒)", flush=True)

    print(f"  完了 ({time.time()-t0:.1f}秒)")

    # 役別の出現統計を表示する
    print("\n  【役別配当統計】")
    for pay in sorted(pay_counts.keys(), reverse=True):
        cnt = pay_counts[pay]
        print(f"    配当{pay:>4}: {cnt:>10,}手 ({cnt/N*100:.4f}%)")

    return table

# ==============================================================================
# GPUカーネル（並列処理の本体）
# ==============================================================================

@cuda.jit
def kernel_giant(initial_hands, deck_indices, giant_table,
                 results_ev, results_hit, results_draw_count,
                 debug_draw5_ev, debug_best_draw):
    """
    GPUカーネル: 各初期手札の最適期待値を計算する。

    【GPUの並列処理とは】

      CPUは数個〜数十個のコアで順番に処理するのに対し、
      GPUは数千個の小さなコア（スレッド）が同時に動く。

      このカーネルは「1スレッド = 1初期手札」で動作する。
      2,598,960手をGPUの数千スレッドが一斉に処理するので高速になる。

    【引数】
      initial_hands : 全初期手札の配列 [N, 5]
                      例: initial_hands[42] = [3, 17, 28, 40, 51]（42番目の手札）
      deck_indices  : 各手札に対応する「残りデッキ47枚」の配列 [N, 47]
                      引き直す候補カードがここに入っている
      giant_table   : 巨大ルックアップテーブル [52^5]
                      table[k] = 手札インデックスkの配当
      results_ev    : 出力: 各手札の最適期待値 [N]
      debug_draw5_ev: 出力: 全捨て時の期待値 [N]（デバッグ用）
      debug_best_draw: 出力: 最適なdraw枚数 [N]（デバッグ用）

    【処理の流れ】

      1. 自分が担当する手札番号 idx を取得する
      2. 32パターン（= 2^5、どのカードを残すかの全組み合わせ）を試す
      3. 各パターンで「引き直し後の期待値」を計算する
      4. 最も期待値の高いパターンの値を results_ev[idx] に書き込む
    """

    # --- 自分が担当する手札のインデックスを取得 ---
    # cuda.grid(1) は「全スレッドの中で自分が何番目か」を返す
    idx = cuda.grid(1)
    if idx >= initial_hands.shape[0]:
        # 手札数がスレッド数で割り切れない場合、余分なスレッドは何もしない
        return

    # --- 担当手札と残りデッキを GPU のローカルメモリに読み込む ---
    # GPU のローカルメモリはレジスタに近い最速のメモリ
    hand = cuda.local.array(5, dtype=np.int32)
    for i in range(5):
        hand[i] = initial_hands[idx, i]

    rem = cuda.local.array(47, dtype=np.int32)
    for i in range(47):
        rem[i] = deck_indices[idx, i]
        # rem は昇順（小さい番号から大きい番号へ）に並んでいる
        # これにより draw=5（全捨て）でネストループが自動的にソート済みになる

    # テーブル参照に使う 52 の累乗（定数）
    P4 = np.int64(52*52*52*52)   # 52⁴ = 7,311,616
    P3 = np.int64(52*52*52)      # 52³ = 140,608
    P2 = np.int64(52*52)         # 52² = 2,704
    P1 = np.int64(52)            # 52¹ = 52

    # 最良の期待値とそのときの draw 枚数を初期化
    best_ev         = -1.0
    best_hit        = 0.0   # 最良パターンのヒット率
    best_draw       = np.int32(0)
    total_draw_count = np.int64(0)  # 全32パターンで検証したDRAW総組み合わせ数

    # ===========================================================
    # 32パターン（どのカードを残すか）を全て試す
    # ===========================================================
    #
    # mask は 0〜31 の整数で、5ビットで「残すカード」を表す。
    # ビット i が立っていれば i 番目のカードを残す。
    #
    # 例: mask = 0b10110 = 22 なら、1番・2番・4番のカードを残す
    #
    # mask=0  → 5枚全部捨てる（draw=5）
    # mask=31 → 5枚全部残す（draw=0）

    for mask in range(32):

        # 残すカード（held）を取り出す
        held = cuda.local.array(5, dtype=np.int32)
        hc = np.int32(0)  # 残す枚数カウンタ
        for i in range(5):
            if mask & (1 << i):
                held[hc] = hand[i]
                hc += 1

        # 引く枚数 = 5 - 残す枚数
        draw = np.int32(5) - hc

        # ===========================================================
        # draw=0: 全枚数をホールド（引き直しなし）
        # ===========================================================
        if draw == 0:
            # held の5枚でそのままテーブルを引く
            k = (np.int64(held[0])*P4 + np.int64(held[1])*P3 +
                 np.int64(held[2])*P2 + np.int64(held[3])*P1 + np.int64(held[4]))
            pay = giant_table[k]
            ev  = float(pay)
            hit = float(np.int32(pay > 0))
            total_draw_count += np.int64(1)  # draw=0は1通りのみ

        # ===========================================================
        # draw=1: 1枚引き直す → C(47,1) = 47通りを全列挙
        # ===========================================================
        elif draw == 1:
            total = np.int64(0)
            hits  = np.int64(0)
            for a in range(47):
                # held の (hc) 番目の位置に rem[a] を差し込む
                # 例: hc=2 なら held[0],held[1] はそのまま、held[2] に rem[a]
                f0=held[0]; f1=held[1]; f2=held[2]; f3=held[3]; f4=held[4]
                if   hc == 1: f1 = rem[a]
                elif hc == 2: f2 = rem[a]
                elif hc == 3: f3 = rem[a]
                else:         f4 = rem[a]
                k   = np.int64(f0)*P4 + np.int64(f1)*P3 + np.int64(f2)*P2 + np.int64(f3)*P1 + np.int64(f4)
                pay = giant_table[k]
                total += pay
                hits  += np.int64(pay > 0)
            # 47通りの配当の合計 ÷ 47 = 平均配当（= 期待値）
            ev  = float(total) / 47.0
            hit = float(hits)  / 47.0
            total_draw_count += np.int64(47)

        # ===========================================================
        # draw=2: 2枚引き直す → C(47,2) = 1,081通りを全列挙
        # ===========================================================
        elif draw == 2:
            total = np.int64(0)
            hits  = np.int64(0)
            for a in range(47):
                for b in range(a+1, 47):
                    # a<b となるよう二重ループを回す（組み合わせ列挙）
                    f0=held[0]; f1=held[1]; f2=held[2]; f3=held[3]; f4=held[4]
                    if   hc == 0: f0=rem[a]; f1=rem[b]   # 0枚残し: 最初の2スロットに
                    elif hc == 1: f1=rem[a]; f2=rem[b]   # 1枚残し: 2・3スロットに
                    elif hc == 2: f2=rem[a]; f3=rem[b]   # 2枚残し: 3・4スロットに
                    else:         f3=rem[a]; f4=rem[b]   # 3枚残し: 4・5スロットに
                    k   = np.int64(f0)*P4 + np.int64(f1)*P3 + np.int64(f2)*P2 + np.int64(f3)*P1 + np.int64(f4)
                    pay = giant_table[k]
                    total += pay
                    hits  += np.int64(pay > 0)
            ev  = float(total) / 1081.0   # C(47,2) = 47×46÷2 = 1,081
            hit = float(hits)  / 1081.0
            total_draw_count += np.int64(1081)

        # ===========================================================
        # draw=3: 3枚引き直す → C(47,3) = 16,215通りを全列挙
        # ===========================================================
        elif draw == 3:
            total = np.int64(0)
            hits  = np.int64(0)
            for a in range(47):
                for b in range(a+1, 47):
                    for c in range(b+1, 47):
                        f0=held[0]; f1=held[1]; f2=held[2]; f3=held[3]; f4=held[4]
                        if   hc == 0: f0=rem[a]; f1=rem[b]; f2=rem[c]
                        elif hc == 1: f1=rem[a]; f2=rem[b]; f3=rem[c]
                        else:         f2=rem[a]; f3=rem[b]; f4=rem[c]
                        k   = np.int64(f0)*P4 + np.int64(f1)*P3 + np.int64(f2)*P2 + np.int64(f3)*P1 + np.int64(f4)
                        pay = giant_table[k]
                        total += pay
                        hits  += np.int64(pay > 0)
            ev  = float(total) / 16215.0  # C(47,3) = 47×46×45÷6 = 16,215
            hit = float(hits)  / 16215.0
            total_draw_count += np.int64(16215)

        # ===========================================================
        # draw=4: 4枚引き直す → C(47,4) = 178,365通りを全列挙
        # ===========================================================
        elif draw == 4:
            total = np.int64(0)
            hits  = np.int64(0)
            for a in range(47):
                for b in range(a+1, 47):
                    for c in range(b+1, 47):
                        for d in range(c+1, 47):
                            f0=held[0]; f1=held[1]; f2=held[2]; f3=held[3]; f4=held[4]
                            if hc == 0: f0=rem[a]; f1=rem[b]; f2=rem[c]; f3=rem[d]
                            else:       f1=rem[a]; f2=rem[b]; f3=rem[c]; f4=rem[d]
                            k   = np.int64(f0)*P4 + np.int64(f1)*P3 + np.int64(f2)*P2 + np.int64(f3)*P1 + np.int64(f4)
                            pay = giant_table[k]
                            total += pay
                            hits  += np.int64(pay > 0)
            ev  = float(total) / 178365.0  # C(47,4) = 178,365
            hit = float(hits)  / 178365.0
            total_draw_count += np.int64(178365)

        # ===========================================================
        # draw=5: 全部捨てて引き直す → C(47,5) = 1,533,939通りを全列挙
        # ===========================================================
        else:
            total = np.int64(0)
            hits  = np.int64(0)
            for a in range(47):
                for b in range(a+1, 47):
                    for c in range(b+1, 47):
                        for d in range(c+1, 47):
                            for e in range(d+1, 47):
                                # rem は昇順に並んでいるので、
                                # a<b<c<d<e の順に取り出せば自動的に順序付きになる。
                                # テーブルは全順列を持っているので順序は問わないが、
                                # ここでは昇順のまま渡す。
                                k   = (np.int64(rem[a])*P4 + np.int64(rem[b])*P3 +
                                       np.int64(rem[c])*P2 + np.int64(rem[d])*P1 + np.int64(rem[e]))
                                pay = giant_table[k]
                                total += pay
                                hits  += np.int64(pay > 0)
            ev  = float(total) / 1533939.0  # C(47,5) = 1,533,939
            hit = float(hits)  / 1533939.0
            total_draw_count += np.int64(1533939)
            debug_draw5_ev[idx] = ev        # 「全捨て」の期待値を記録（デバッグ用）

        # ===========================================================
        # このパターンの期待値が今までの最大を超えたら更新する
        # ===========================================================
        if ev > best_ev:
            best_ev   = ev
            best_hit  = hit
            best_draw = draw

    # 32パターンの中で最も高かった期待値・HIT率を出力配列に書き込む
    results_ev[idx]         = best_ev
    results_hit[idx]        = best_hit
    results_draw_count[idx] = total_draw_count
    debug_best_draw[idx]    = best_draw

# ==============================================================================
# メイン処理
# ==============================================================================

def main():
    print("=" * 65)
    print("Jacks or Better 還元率計算（巨大ルックアップテーブル版）")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # [1/5] 初期手札と残りデッキを生成する
    # -----------------------------------------------------------------------
    print("\n[1/5] 初期手札・残りデッキを生成中...")
    t0 = time.time()

    deck  = list(range(DECK_SIZE))  # [0, 1, 2, ..., 51]
    hands = list(combinations(deck, 5))  # C(52,5) = 2,598,960通りの組み合わせ
    N     = len(hands)

    # NumPy配列に変換（GPUへの転送に必要）
    hands_np = np.array(hands, dtype=np.int32)  # shape: [2,598,960, 5]

    # 各手札に対応する「残りデッキ47枚」を計算する
    # （手札の5枚を除いた残り47枚が引き直しの候補になる）
    remaining_np = np.zeros((N, 47), dtype=np.int32)
    for i, hand in enumerate(hands):
        hand_set = set(hand)
        # デッキ全52枚から手札5枚を除いた47枚を昇順で格納する
        remaining_np[i] = [c for c in range(DECK_SIZE) if c not in hand_set]

    print(f"  総手数: {N:,}  完了 ({time.time()-t0:.1f}秒)")

    # -----------------------------------------------------------------------
    # [2/5] 巨大ルックアップテーブルを構築する
    # -----------------------------------------------------------------------
    print("\n[2/5] 巨大ルックアップテーブルを構築中...")
    t0 = time.time()
    giant_table = build_giant_table(hands)
    print(f"  テーブル構築完了 ({time.time()-t0:.1f}秒)")

    # -----------------------------------------------------------------------
    # [3/5] データをGPUに転送する
    # -----------------------------------------------------------------------
    # GPU（グラフィックカード）は CPU とは別のメモリを持っているため、
    # 計算に使うデータを事前に GPU のメモリ（VRAM）にコピーする必要がある。
    # この転送は一度だけ行い、GPU 計算中は転送しない（高速化のため）。
    print("\n[3/5] GPUへ転送中...")
    t0 = time.time()
    vram_gb = TABLE_SIZE * 2 / 1024**3
    print(f"  巨大テーブル転送: {vram_gb:.2f} GB")
    try:
        d_table = cuda.to_device(giant_table)   # テーブル（約726MB）をVRAMへ
    except Exception as e:
        print(f"\n  [エラー] GPU転送失敗: {e}")
        print("  VRAMが不足している可能性があります。")
        return
    d_hands = cuda.to_device(hands_np)          # 手札データをVRAMへ
    d_rem   = cuda.to_device(remaining_np)       # 残りデッキデータをVRAMへ
    print(f"  完了 ({time.time()-t0:.1f}秒)")

    # -----------------------------------------------------------------------
    # [4/5] GPUカーネルを実行する
    # -----------------------------------------------------------------------
    # TDR（GPU タイムアウト）を避けるため、全手札を BATCH 件ずつに分割して
    # カーネルを繰り返し呼び出す。
    # （Windows では GPU が長時間応答しないとシステムが強制リセットする）
    print("\n[4/5] GPUカーネル実行中...")
    t0 = time.time()

    threads = 64     # 1ブロックあたりのスレッド数（GPU の並列単位）
    BATCH   = 10000  # 1回のカーネル呼び出しで処理する手数
    num_batches = math.ceil(N / BATCH)

    # 結果を格納する配列（CPU側）
    results_all        = np.zeros(N, dtype=np.float64)  # 各手の最適期待値
    results_hit        = np.zeros(N, dtype=np.float64)  # 各手の最適HIT率
    results_draw_count = np.zeros(N, dtype=np.int64)    # 各手で検証したDRAW総組み合わせ数
    draw5_ev_all       = np.zeros(N, dtype=np.float64)  # 各手の全捨て期待値（デバッグ用）
    best_draw_all      = np.zeros(N, dtype=np.int32)    # 各手の最適 draw 枚数（デバッグ用）

    for bi in range(num_batches):
        s  = bi * BATCH
        e  = min(s + BATCH, N)
        sz = e - s

        # このバッチ分のデータを GPU に転送する
        bh  = cuda.to_device(hands_np[s:e])
        br  = cuda.to_device(remaining_np[s:e])

        # GPU上に結果用の空配列を確保する
        res = cuda.device_array(sz, dtype=np.float64)
        rht = cuda.device_array(sz, dtype=np.float64)
        rdc = cuda.device_array(sz, dtype=np.int64)
        d5  = cuda.device_array(sz, dtype=np.float64)
        bd  = cuda.device_array(sz, dtype=np.int32)

        # カーネルを起動する
        # 書式: カーネル関数[ブロック数, スレッド数/ブロック](引数...)
        blk = math.ceil(sz / threads)
        kernel_giant[blk, threads](bh, br, d_table, res, rht, rdc, d5, bd)

        # GPU の処理が完了するまで待つ
        cuda.synchronize()

        # 結果を GPU から CPU にコピーして配列に格納する
        results_all[s:e]        = res.copy_to_host()
        results_hit[s:e]        = rht.copy_to_host()
        results_draw_count[s:e] = rdc.copy_to_host()
        draw5_ev_all[s:e]       = d5.copy_to_host()
        best_draw_all[s:e]      = bd.copy_to_host()

        # 進捗を表示する
        elapsed = time.time() - t0
        pct = e / N * 100
        eta = elapsed / (pct / 100) * (1 - pct / 100) if pct > 0 else 0
        print(f"\r  進捗: {pct:5.1f}% ({e:,}/{N:,}) "
              f"経過: {elapsed:.0f}秒 残り: {eta:.0f}秒",
              end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  完了 ({elapsed:.1f}秒)")

    # -----------------------------------------------------------------------
    # [5/5] 結果を集計して表示する
    # -----------------------------------------------------------------------
    print("\n[5/5] 集計・デバッグ出力...")

    # RTP = 全手札の最適期待値の平均
    # （「1コイン賭けたとき平均何コイン返ってくるか」= 期待値の平均）
    rtp = results_all.mean()
    hit = results_hit.mean()
    total_draw_count_sum = results_draw_count.sum()

    print("\n" + "=" * 65)
    print(f"  推定還元率 (RTP) : {rtp * 100:.4f}%")
    print(f"  ヒット率   (Hit) : {hit * 100:.4f}%  (約 1/{1/hit:.1f} スピンに1回)")
    print(f"  DRAW検証回数総和 : {total_draw_count_sum:,}")
    print("=" * 65)

    print("\n【デバッグ情報】")
    # 全捨てEV: どの手でも全部捨てた場合の平均期待値（最適戦略と比較する基準）
    print(f"  全捨てEV平均      : {draw5_ev_all.mean():.6f}")
    print(f"  最終EV平均        : {results_all.mean():.6f}")
    print(f"  最終EV最大値      : {results_all.max():.4f}")
    print(f"  最終EV最小値      : {results_all.min():.4f}")
    print(f"  EV>10の手数       : {(results_all > 10).sum():,}")
    print(f"  EV>100の手数      : {(results_all > 100).sum():,}")
    print(f"  EV>500の手数      : {(results_all > 500).sum():,}")

    # 最適戦略において何枚引き直すかの分布
    print("\n【最適draw数の分布】")
    draw_counts = Counter(best_draw_all.tolist())
    for d in sorted(draw_counts.keys()):
        cnt = draw_counts[d]
        mask_results = results_all[best_draw_all == d]
        print(f"  draw={d}: {cnt:>10,}手 ({cnt/N*100:.2f}%) "
              f"平均EV={mask_results.mean():.4f}")

    # 期待値の高い上位10手を表示する
    print("\n【EV上位10手】")
    top10_idx = np.argsort(results_all)[-10:][::-1]
    for i in top10_idx:
        hand_str = " ".join(card_name(c) for c in hands[i])
        print(f"  [{hand_str}] EV={results_all[i]:.4f} draw={best_draw_all[i]}")


if __name__ == "__main__":
    main()
