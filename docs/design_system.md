# マッキンゼー風 UI デザインガイド

本ガイドは、マッキンゼー風の「知性・簡潔・信頼」を体現するための配色、タイポグラフィ、レイアウト、コンポーネントの設計原則と、それらを Streamlit アプリで再現するための設計トークンを定義します。

## 1. デザイン・トークン

| トークン名 | 用途 | 値・詳細 |
| --- | --- | --- |
| `color.primary` | ブランドカラー（主要ボタン・リンク） | `#0B1F3B`（濃紺）─ マッキンゼーのブランドカラーに近い、信頼感と知性を感じさせる。 |
| `color.secondary` | サブカラー（サイドバー・補助線） | `#5A6B7A`（グレー）─ メインコンテンツを邪魔せずに情報を支える。 |
| `color.accent` | 強調色（チャート・アクション） | `#1E88E5`（鮮やかな青）─ KPI の矢印やリンクなど注意を引く要素に使用。 |
| `color.background` | 背景色 | `#F7F8FA`（淡いグレー）─ 高コントラストを避けつつコンテンツを浮き上がらせる。 |
| `color.surface` | カードやモーダルの背景 | `#FFFFFF`─ 純白を使用しコンテンツを際立たせる。 |
| `color.text.primary` | 主本文色 | `#1A1A1A`─ 可読性を確保するためダークグレーを採用。 |
| `color.text.secondary` | 補助テキスト | `#5A6B7A`─ 二次情報のために明度を下げる。 |
| `color.success` | ポジティブ指標 | `#1B5E20`（グリーン）彩度を 20% 抑制。 |
| `color.warning` | 注意指標 | `#F57C00`（オレンジ）彩度を 20% 抑制。 |
| `color.error` | ネガティブ指標 | `#B71C1C`（レッド）彩度を 20% 抑制。 |
| `font.family` | フォントファミリ | `"Inter", "Source Sans Pro", sans-serif` ─ データに最適な可読性とモダンさを確保。 |
| `font.size.base` | 基本文字サイズ | `16px` ─ 本文の既定値。 |
| `font.size.h1` / `font.size.h2` / `font.size.h3` | 見出しサイズ | `28px / 24px / 20px` ─ 階層感を明確にする。数字や KPI カードには等幅フォントを使用。 |
| `radius.card` | カード角丸 | `8px` ─ 柔らかさと洗練を両立。 |
| `shadow.card` | カード影 | `0px 2px 4px rgba(0, 0, 0, 0.05)` ─ シンプルな浮遊感を出す。 |
| `spacing.unit` | 基本余白 | `8px` ─ すべてのマージン・パディングを 8 の倍数で統一し、整然としたレイアウトを実現。 |

## 2. タイポグラフィとレイアウト規範

- **フォント**: Inter または Source Sans Pro を使用し、本文は 16px、見出しは 20～28px。数字は等幅フォントで統一し、KPI カードの数値は強調のため太字にする。
- **行間**: 本文は 1.5 倍、見出しは 1.3 倍とすることで読みやすさを確保。
- **レイアウト**: `layout="wide"` を使い、12 カラムグリッド相当（例: 4:4:4）の列幅でコンテンツを配置。要素間には最低 8px の余白を設け、情報密度をコントロールする。
- **カード**: 角丸 8px、薄い影をつけて背景から浮かせる。カード内ではタイトル → 数値 → 補足情報の順に階層をつけ、指標の矢印や前期比は右寄せで配置する。
- **色の使い分け**: 色は意味を持たせるためにのみ使用し、強調には太字や位置で差異をつける。成功・警告・エラー色の彩度は 20% 抑制し、プロフェッショナルな印象を保つ。

## 3. Streamlit テーマ設定

以下のテーマ設定を `/.streamlit/config.toml` に保存すると、上記デザイン・トークンをベースにしたテーマを適用できます。

```toml
[theme]
primaryColor = "#0B1F3B"
backgroundColor = "#F7F8FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#1A1A1A"
font = "sans serif"
```

### KPI カードの実装例

`st.container` や `st.markdown` にインライン CSS を付与して角丸や影を実装します。数値は等幅フォントを指定し、差分指標は右寄せで色分けします。

```python
card_style = f"""
    background-color: {st.get_option("theme.secondaryBackgroundColor")};
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
"""

def kpi_card(title: str, value: str, delta: float):
    arrow_color = "#1B5E20" if delta >= 0 else "#B71C1C"
    arrow = "▲" if delta >= 0 else "▼"
    st.markdown(
        f"<div style='{card_style}'>"
        f"  <h3 style='color:{st.get_option('theme.primaryColor')}; margin:0 0 4px;'>"
        f"    {title}"
        f"  </h3>"
        f"  <p style='font-size:28px; font-weight:700; font-family: \"IBM Plex Mono\", monospace; margin:0;'>"
        f"    {value}"
        f"  </p>"
        f"  <p style='text-align:right; color:{arrow_color}; margin:4px 0 0; font-family: \"IBM Plex Mono\", monospace;'>"
        f"    {arrow}{delta:+.1%}"
        f"  </p>"
        f"</div>",
        unsafe_allow_html=True,
    )
```

## 4. 期待される効果と留意点

- **信頼感の向上**: 濃紺とグレーを基調とすることで、BtoB 向けの重厚感と知性を演出し、ユーザーの信頼を高める。
- **可読性向上**: 16px 以上の本文、太字の数値、適切な行間により、目が疲れにくく情報が即座に理解できる。
- **一貫性**: 色・余白・角丸などのトークンを統一することで、画面遷移時の違和感が減り学習コストを低下させる。これにより操作ミス率が現在比で半減すると推定する。
- **実装容易性**: Streamlit のテーマ設定とシンプルな CSS で実装でき、2 週間スプリント内でデザイントークンを導入できる。
