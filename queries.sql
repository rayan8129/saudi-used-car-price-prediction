-- Saudi Used Car Price Prediction: SQL analysis (SQLite)
-- Run through sql_analysis.py (or paste into any SQLite client). Tables: cars_raw, model_predictions. View: cars_clean.

-- Q1. Data quality: how many raw listings have no asking price?
--     (Negotiable is stored as 1/0 after loading TRUE/FALSE with pandas)
SELECT COUNT(*)                                                  AS total_listings,
       SUM(Price = 0)                                            AS zero_price,
       ROUND(100.0 * SUM(Price = 0) / COUNT(*), 1)               AS zero_price_pct,
       SUM(Price = 0 AND Negotiable = 1)                    AS zero_price_and_negotiable
FROM cars_raw;

-- Q2. Market overview by make (makes with 50+ listings): count, average price, mileage
SELECT Make,
       COUNT(*)                   AS listings,
       ROUND(AVG(Price))          AS avg_price,
       ROUND(AVG(Mileage))        AS avg_mileage
FROM cars_clean
GROUP BY Make
HAVING COUNT(*) >= 50
ORDER BY avg_price DESC;

-- Q3. Average price by region
SELECT Region,
       COUNT(*)          AS listings,
       ROUND(AVG(Price)) AS avg_price
FROM cars_clean
GROUP BY Region
ORDER BY listings DESC;

-- Q4. Depreciation: average price by car age bucket
SELECT CASE WHEN Car_Age <= 2  THEN '1) 0-2 years'
            WHEN Car_Age <= 5  THEN '2) 3-5 years'
            WHEN Car_Age <= 10 THEN '3) 6-10 years'
            ELSE                    '4) 11+ years' END AS age_bucket,
       COUNT(*)          AS listings,
       ROUND(AVG(Price)) AS avg_price
FROM cars_clean
GROUP BY age_bucket
ORDER BY age_bucket;

-- Q5. Does the option level matter?
SELECT Options,
       COUNT(*)          AS listings,
       ROUND(AVG(Price)) AS avg_price
FROM cars_clean
GROUP BY Options
ORDER BY avg_price DESC;

-- Q6. Price premium of each make over its region's overall average (JOIN on a sub-query)
SELECT c.Region, c.Make,
       COUNT(*)                          AS listings,
       ROUND(AVG(c.Price))               AS make_avg_price,
       ROUND(r.region_avg)               AS region_avg_price,
       ROUND(AVG(c.Price) - r.region_avg) AS premium_vs_region
FROM cars_clean c
JOIN (SELECT Region, AVG(Price) AS region_avg FROM cars_clean GROUP BY Region) r
  ON r.Region = c.Region
WHERE c.Region IN ('Riyadh', 'Jeddah', 'Dammam')
GROUP BY c.Region, c.Make
HAVING COUNT(*) >= 30
ORDER BY c.Region, premium_vs_region DESC;

-- Q7. Top 3 most expensive makes inside each region (CTE + RANK)
WITH make_region AS (
    SELECT Region, Make, COUNT(*) AS listings, ROUND(AVG(Price)) AS avg_price
    FROM cars_clean
    GROUP BY Region, Make
    HAVING COUNT(*) >= 20
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY Region ORDER BY avg_price DESC) AS rnk
    FROM make_region
)
SELECT Region, rnk, Make, listings, avg_price
FROM ranked
WHERE rnk <= 3
ORDER BY Region, rnk;

-- Q8. Model performance: accuracy bands recomputed from Pct_Error
--     (8 test cars with error above ~1,800% have a blank band in the CSV, so the
--      bands are rebuilt here with CASE instead of using Prediction_Accuracy_Band)
SELECT CASE WHEN Pct_Error < 10 THEN '1) Excellent (<10%)'
            WHEN Pct_Error < 25 THEN '2) Good (10-25%)'
            WHEN Pct_Error < 50 THEN '3) Fair (25-50%)'
            ELSE                     '4) Poor (>50%)' END AS accuracy_band,
       COUNT(*)                                           AS cars,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_test_set
FROM model_predictions
GROUP BY accuracy_band
ORDER BY accuracy_band;

-- Q9. Where does the model struggle? Error by make (makes with 10+ test cars)
SELECT Make,
       COUNT(*)                AS test_cars,
       ROUND(AVG(Abs_Error))   AS mean_abs_error
FROM model_predictions
GROUP BY Make
HAVING COUNT(*) >= 10
ORDER BY mean_abs_error DESC
LIMIT 10;

-- Q10. Does the model over- or under-predict? (bias by price segment)
SELECT CASE WHEN True_Price < 40000  THEN '1) under 40K'
            WHEN True_Price < 100000 THEN '2) 40K-100K'
            ELSE                          '3) 100K+' END AS price_segment,
       COUNT(*)                                          AS cars,
       ROUND(AVG(Predicted_Price - True_Price))          AS avg_signed_error,
       ROUND(AVG(Pct_Error),1)                           AS mean_pct_error
FROM model_predictions
GROUP BY price_segment
ORDER BY price_segment;

-- Q11. Join: model error vs. the market's average price for the same make
SELECT p.Make,
       COUNT(*)                 AS test_cars,
       ROUND(m.avg_price)       AS market_avg_price,
       ROUND(AVG(p.Abs_Error))  AS mean_abs_error
FROM model_predictions p
JOIN (SELECT Make, AVG(Price) AS avg_price FROM cars_clean GROUP BY Make) m
  ON m.Make = p.Make
GROUP BY p.Make
HAVING COUNT(*) >= 10
ORDER BY market_avg_price DESC
LIMIT 10;
