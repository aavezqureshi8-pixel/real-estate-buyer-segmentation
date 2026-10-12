# Viva Questions and Answers

Answers are specific to this implementation and use the real results of the analysis.

**Q1. What is the aim of the project?**

To find hidden groups of real-estate buyers and describe each group's investment behaviour, so that marketing and property recommendations can be targeted instead of generic.

**Q2. Which datasets did you use and how are they connected?**

`clients.csv` (2,000 clients) and `properties.csv` (10,000 listings). The key is `clients.client_id` = `properties.client_ref`. I discovered it by testing which column's values all exist in a unique column of the other file (100.0% match). Only sold listings have a client_ref.

**Q3. Why is the unit of analysis the client and not the property?**

Segmentation is about buyers. A client owns several units (3.65 on average), so I aggregated the sold units per client into features such as units owned, average price and office share.

**Q4. What data problems did you find?**

Three different date-of-birth layouts, prices stored as text like `$300,385.62`, and the label `Company` instead of the PRD's `Corporate`. There were no missing values and no duplicates in the clients file; missing `client_ref` values belong to unsold listings, which is normal.

**Q5. How did you handle dates of birth?**

Parsed each layout explicitly and converted to age at 2025-12-01 (the last transaction date). Raw dates are not used in clustering. Dash dates could be day-first or month-first; I assumed month-first (matching the transaction dates), which changes age by at most one year.

**Q6. How did you handle outliers?**

I detected them with the IQR rule but did not delete any. Sale price and area had 0 outliers; high-value units are legitimate real-estate sales. The clients who own up to 13 units are kept because they form a real segment.

**Q7. Why did you drop region from the model?**

Every region belongs to exactly one country, so region repeats country, and its 57 one-hot columns would dominate the distance calculation. Region is used for profiling and maps instead.

**Q8. Why not use gender?**

It is split almost 50/50, adds no segmentation value, and targeting by gender is ethically questionable. I only use it to describe the clusters.

**Q9. Which encoding did you use and why?**

Binary variables (purpose, loan, client type) are label-encoded to 0/1. Multi-category nominal variables (country, referral channel) are one-hot encoded, because label encoding would invent an order (for example Canada < France < UK).

**Q10. Why is scaling needed?**

K-Means and Ward use Euclidean distance. Average price is in hundreds of thousands while satisfaction is 1-5; unscaled, price would decide every distance. StandardScaler gives each numeric feature mean 0 and standard deviation 1.

**Q11. What is K-Means?**

It places K centres, assigns each client to the nearest centre, moves each centre to the mean of its clients, and repeats until nothing changes. It minimises within-cluster variance (inertia).

**Q12. What is the elbow method?**

Plot inertia against K. Inertia always falls as K grows, but the gain shrinks after a point - the 'elbow'. I also computed it automatically (largest distance from the straight line joining the end points); it gave K = 6.

**Q13. What does the silhouette score mean?**

For each client: (distance to nearest other cluster - distance to own cluster) / the larger of the two. It lies between -1 and +1; higher is better. Mine is 0.123, which is low, meaning the groups overlap.

**Q14. How did you choose K and why not just K = 4 like the PRD?**

I did not assume it. Elbow method suggests K=6; the unrestricted silhouette peak is K=2 (a trivial 2-way split). Restricting to 3-6 segments (business limit), K values with silhouette within 0.01 of the best (0.124): [3, 4, 5, 6]. Of these, K values with stability ARI >= 0.6: [3, 6]. K=6 has the lowest Davies-Bouldin index (1.696) among them (silhouette=0.123, stability ARI=0.68).

**Q15. Silhouette is only 0.12. Is the project a failure?**

No. It is an honest result: the data has weak natural separation (almost all behavioural variables are independent of each other). I report it, show stability and statistical tests, and use the segments as practical groupings. Forcing artificial separation would be misleading.

**Q16. What is hierarchical clustering and why did you use it?**

It builds a tree by repeatedly merging the two closest clusters; cutting the tree gives clusters. I used Ward linkage (merge that increases variance least) to cross-check K-Means.

**Q17. How do K-Means and hierarchical results compare?**

Silhouette 0.123 (K-Means) vs 0.090 (Ward); agreement ARI = 0.25, so they only partly agree. Both isolate the small high-volume group. Average linkage looked better on silhouette but created tiny degenerate clusters.

**Q18. Why is K-Means your main model?**

Better quality metrics, repeatable results (stability ARI 0.68), fast, and it can assign new clients with `predict()` - which the dashboard predictor uses.

**Q19. What is PCA and why use it?**

PCA compresses the 21 encoded features into 2 axes so clusters can be plotted. It is only for visualisation; clustering used all features. The two axes explain 34.3% of variance, so the picture is a simplified view.

**Q20. What does each cluster represent?**

C1 = Office-Focused Senior Buyers (437 clients). C2 = Senior Buyers (415 clients). C3 = Highly Satisfied Younger Buyers (397 clients). C4 = Dissatisfied Younger Buyers (387 clients). C5 = Premium-Ticket Younger Buyers (313 clients). C6 = High-Volume Senior Buyers (51 clients).

**Q21. How did you name the clusters?**

Automatically from the data: for each cluster I calculated how many standard deviations it differs from the overall average on each trait, and used the two strongest traits (|effect| ≥ 0.30). So names describe real differences.

**Q22. Do the PRD segments (Global Investors, etc.) exist?**

None of them fully meets its conditions. Example: no cluster has both a clearly higher investment share and a higher international share than average; 'first-time' and 'income' are not in the data.

**Q23. Which clusters are most valuable?**

By value index (revenue share / client share), C6 (High-Volume Senior Buyers) (1.90). By total investor spend, C5 (Premium-Ticket Younger Buyers).

**Q24. Which buyers are investment-oriented, and which depend on loans?**

Investment rate is 30.8% overall; the highest is C6 (High-Volume Senior Buyers) at 35.3%. Loan use is 36.8% overall; highest in C3 (Highly Satisfied Younger Buyers) (41.6%). The chi-square tests show these differences between clusters are not statistically significant.

**Q25. Did you test whether the clusters really differ?**

Yes: chi-square (categorical) and Kruskal-Wallis (numeric) tests with effect sizes. Numeric differences are significant by construction; purpose, loan, channel, country and gender are not significant.

**Q26. How would you avoid data leakage?**

There is no target variable in unsupervised clustering. I avoided features that merely restate other features (total spend = units x price) and listing-level identifiers. The preprocessor is fitted once and re-used for new clients.

**Q27. What are the limitations?**

Weak cluster separation, no income/loan amount/first-time flag, uniform-looking data, small countries, month-first date assumption, and untested business impact of recommendations.

**Q28. What is the business value?**

Different budgets, ages, portfolio sizes and satisfaction levels call for different messages, inventory and service. Examples: retention for the low-satisfaction group, key-account service for high-volume buyers, premium stock for high-ticket buyers.

**Q29. How does the dashboard work?**

`train_pipeline.py` trains and saves everything once; `app.py` (Streamlit + Plotly) only loads the saved files, applies sidebar filters (country, region, purpose, client type, age, price, cluster, loan) and redraws charts and KPIs. A predictor page uses the saved models on a new client.

**Q30. How would you improve the project?**

Add income, loan amounts and purchase history; test K-Prototypes or Gaussian mixtures for mixed data; validate segments through A/B campaigns.

