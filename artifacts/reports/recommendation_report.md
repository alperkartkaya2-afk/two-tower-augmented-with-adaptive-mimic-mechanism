# Recommendation Evaluation Report

## Ranking Metrics

- **Recall**: @5=0.0709, @10=0.1001, @20=0.1393
- **Precision**: @5=0.0142, @10=0.0100, @20=0.0070
- **NDCG**: @5=0.0503, @10=0.0597, @20=0.0696
- **Hit Rate**: @5=0.0709, @10=0.1001, @20=0.1393
- **MAP**: @5=0.0436, @10=0.0474, @20=0.0501

## Loss Curves

Training, validation, and test losses tracked across epochs. Monitoring metric:
- Best recall@10 achieved at epoch 7
![Loss curves](artifacts/reports/loss_curve.png)

Epoch | Train | Validation | Test
--- | --- | --- | ---
1 | 0.3754 | 0.2853 | 0.2852
2 | 0.2855 | 0.2496 | 0.2468
3 | 0.2582 | 0.2379 | 0.2336
4 | 0.2433 | 0.2352 | 0.2291
5 | 0.2320 | 0.2369 | 0.2275
6 | 0.2241 | 0.2482 | 0.2318
7 | 0.2192 | 0.2541 | 0.2327

## Embedding Diagnostics

- User embedding norms: mean=7.4504, std=1.5896, min=4.5270, max=12.5406
- Item embedding norms: mean=3.1077, std=0.5801, min=2.0456, max=4.1149
- Item neighbor category overlap (k=5): mean=0.2000, std=0.1549
- User embedding vs. feature alignment (cosine): mean=0.9207, std=0.0639

### Feature Correlations

Feature | Pearson r | p-value
--- | --- | ---
numeric:average_rating | -0.6618 | 3.71e-02
category:Children's Books > Activities, Crafts & Games | 0.5787 | 7.96e-02
text:title_char_count | 0.5450 | 1.03e-01
category:Mystery, Thriller & Suspense | -0.5261 | 1.18e-01
category:Mystery, Thriller & Suspense > Thrillers & Suspense | -0.5261 | 1.18e-01
numeric:rating_number | -0.4712 | 1.69e-01
text:title_word_count | 0.4243 | 2.22e-01
category:Cookbooks, Food & Wine | 0.3976 | 2.55e-01
category:Cookbooks, Food & Wine > Beverages & Wine | 0.3976 | 2.55e-01
category:Children's Books > Animals | -0.3067 | 3.89e-01
category:Humor & Entertainment | 0.2691 | 4.52e-01
category:Humor & Entertainment > Puzzles & Games | 0.2691 | 4.52e-01
author:Brad Meltzer | 0.2691 | 4.52e-01
numeric:price | 0.2676 | 4.55e-01
category:Children's Books | 0.2040 | 5.72e-01

## Sample User Recommendations

- **User** `AEXEVSIMLCGGHWXIKAGS3IASRWWA` | category match 80.00% | author match 0.00%
  - Historical categories: Humor & Entertainment, Humor & Entertainment > Television, Mystery, Thriller & Suspense, Mystery, Thriller & Suspense > Mystery, Mystery, Thriller & Suspense > Thrillers & Suspense
  1. ["The Atlantis Gene: A Thriller (The Origin Mystery, Book 1)"] (B00C2WDD5I) — author: A.G. Riddle | categories: Literature & Fiction, Literature & Fiction > Action & Adventure
  2. ["The Butterfly Garden (The Collector Book 1)"] (B016ZNRC0Q) — author: Dot Hutchison | categories: Mystery, Thriller & Suspense, Mystery, Thriller & Suspense > Thrillers & Suspense
  3. ["Brilliance (The Brilliance Trilogy Book 1)"] (B00AESRRQS) — author: Marcus Sakey | categories: Mystery, Thriller & Suspense, Mystery, Thriller & Suspense > Thrillers & Suspense
  4. ["Trouble in Mudbug (Ghost-in-Law Mystery/Romance Book 1)"] (B004AYDC9I) — author: Jana DeLeon | categories: Mystery, Thriller & Suspense, Mystery, Thriller & Suspense > Mystery
  5. ["I Am Watching You"] (B06Y1264PX) — author: Teresa Driscoll | categories: Mystery, Thriller & Suspense, Mystery, Thriller & Suspense > Thrillers & Suspense

- **User** `AEZPKBNTC64MKM3Q62LHT62DDORQ` | category match 80.00% | author match 20.00%
  - Historical categories: Children's Books, Children's Books > Growing Up & Facts of Life, Christian Books & Bibles, Christian Books & Bibles > Christian Living, Cookbooks, Food & Wine
  1. ["The Shack: Where Tragedy Confronts Eternity"] (0964729237) — author: William P. Young | categories: Christian Books & Bibles, Christian Books & Bibles > Literature & Fiction
  2. ["The Life-Changing Magic of Tidying Up: The Japanese Art of Decluttering and Organizing"] (1607747308) — author: Marie Kondo | categories: Crafts, Hobbies & Home, Crafts, Hobbies & Home > Home Improvement & Design
  3. ["The Hunger Games Trilogy Boxed Set"] (0545265355) — author: Suzanne Collins | categories: Teen & Young Adult, Teen & Young Adult > Literature & Fiction
  4. ["Secret Garden: An Inky Treasure Hunt and Coloring Book for Adults"] (1780671067) — author: Johanna Basford | categories: Arts & Photography, Arts & Photography > Graphic Design
  5. ["101 Dog Tricks: Step by Step Activities to Engage, Challenge, and Bond with Your Dog (Volume 1) (Dog Tricks and Training, 1)"] (1592533256) — author: Kyra Sundance | categories: Crafts, Hobbies & Home, Crafts, Hobbies & Home > Pets & Animal Care
