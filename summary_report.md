# Credit Card Customer Segmentation — Summary Report

## Business Problem & Dataset

The goal was to segment credit card customers into behaviourally distinct groups so that a card issuer's marketing and credit-risk teams can target each group with tailored offers (rewards, activation campaigns, EMI conversion, fraud monitoring, etc.) instead of a one-size-fits-all approach. The dataset (`CC_GENERAL.csv`) contains 8,950 customers described by 17 behavioural fields covering balance, purchases (one-off vs. installment), cash advances, purchase/cash-advance frequencies, credit limit, payments, minimum payments, percentage of full payment, and tenure. `CUST_ID` was dropped as a non-informative identifier. Missing values in `CREDIT_LIMIT` (1 record) and `MINIMUM_PAYMENTS` (313 records) were imputed with the column median, and no duplicate rows were found.

## Feature Engineering & Preprocessing

Four behavioural ratios were engineered to capture spending *patterns* rather than raw magnitudes: `Monthly_Avg_Purchase` and `Monthly_Avg_Cash_Advance` (activity normalized by tenure), `Limit_Usage` (balance relative to credit limit), and `Payment_to_Minpayment_Ratio` (repayment discipline). These ratios matter more for segmentation than absolute dollar values because they distinguish, for example, a long-tenured heavy spender from a new customer with proportionally similar behaviour.

Monetary columns (BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS) were heavily right-skewed with extreme outliers, so they were winsorized at Q3 + 3×IQR (capping, not dropping, rows) and then log1p-transformed to symmetrize their distributions. All engineered and transformed features were then standardized with `StandardScaler` so that no single high-variance feature (e.g., CASH_ADVANCE) would dominate distance-based clustering. PCA retaining 95% of variance reduced the 22 features to 15 components for exploratory checks, though clustering itself was run on the full scaled feature set.

## Algorithm Comparison

Three algorithms were compared: K-Means, Agglomerative Clustering (Ward/Complete/Average linkage), and DBSCAN. By raw Silhouette Score, Agglomerative with average linkage scored highest (0.76), followed by DBSCAN (0.57), with K-Means lowest (0.18). However, this ranking is misleading: average linkage produced one giant cluster containing nearly all 8,950 customers plus a handful of singleton outliers — a "chaining" artifact that inflates silhouette but is useless for business segmentation. DBSCAN similarly flagged 95% of customers as noise, leaving only ~430 points across six tiny clusters, again too sparse to act on. **K-Means (k=5) was therefore selected as the best-performing algorithm for business purposes** despite its lower silhouette score, because it produced five well-populated, interpretable segments and proved stable across five random seeds (mean silhouette 0.1805 ± 0.0004). This did *not* match the initial intuition that the highest-silhouette method would win — it highlighted why internal metrics must be checked against cluster size distributions before trusting them.

## Cardholder Segments (K-Means, k=5)

- **Regular Customers (Cluster 0, ~3,294):** Moderate balances and purchases with typical payment behaviour — the largest, "average" segment.
- **Low Usage Customers (Cluster 1, ~2,596):** Low purchases, low balances, and low credit utilization; engaged but under-active.
- **Dormant Cardholders (Cluster 2, ~1,485):** Minimal transaction activity of any kind — effectively inactive accounts.
- **Premium Spenders (Cluster 3, ~547):** High purchases, high balances, and high credit limits — the issuer's most valuable spenders.
- **Cash Advance Users (Cluster 4, ~1,028):** Disproportionately reliant on cash advances rather than purchases, a higher credit-risk profile.

## Next Steps

Future iterations should incorporate demographic and transaction-channel data (age, income, merchant category) to enrich the purely behavioural feature set; explore semi-supervised refinement by seeding clusters with known high-value or high-risk accounts; and productionize the saved K-Means pipeline (`cc_scaler.pkl` + `cc_segmentation_model.pkl`) behind a real-time scoring API so new or updated customer records can be assigned a segment on demand.
