WITH 
transactions AS (
SELECT hits.transaction.transactionId AS ID_Transaction, device.deviceCategory,device.operatingSystem,
trafficSource.campaign, trafficSource.medium, geoNetwork.continent, fullvisitorid , hp.v2ProductName AS Product, 
hp.v2ProductCategory AS Product_Category,IFNULL(SUM(hits.transaction.transactionRevenue/1000000),0) AS CA, 
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20161201*` AS ga, 
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp 
GROUP BY ID_Transaction, device.deviceCategory,device.operatingSystem,trafficSource.campaign, 
trafficSource.medium, geoNetwork.continent, fullvisitorid , Product, Product_Category
ORDER BY CA DESC,ID_Transaction ), 
visits_products AS (  
SELECT fullvisitorid, hp.v2ProductName AS Product, SUM( totals.visits) AS Product_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,UNNEST(ga.hits) AS hits, 
UNNEST(hits.product) AS hp WHERE _TABLE_SUFFIX <= '20161201'
GROUP BY fullvisitorid, Product ORDER BY fullvisitorid, Product_Visits DESC ), 
visits_products_category AS (  
SELECT fullvisitorid, hp.v2ProductCategory AS Product_Category, SUM( totals.visits) Product_Category_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,UNNEST(ga.hits) AS hits, 
UNNEST(hits.product) AS hp WHERE _TABLE_SUFFIX <= '20161201'
GROUP BY fullvisitorid, Product_Category ORDER BY fullvisitorid, Product_Category_Visits DESC )
SELECT ID_Transaction, deviceCategory, operatingSystem, campaign, medium, continent,
transactions.Product, Product_Visits, transactions.Product_Category, Product_Category_Visits, 
CASE WHEN CA = 0 THEN 0 ELSE 1 END AS Transaction
FROM transactions LEFT JOIN visits_products
ON transactions.fullvisitorid = visits_products.fullvisitorid AND transactions.Product = visits_products.Product
LEFT JOIN visits_products_category
ON transactions.fullvisitorid = visits_products_category.fullvisitorid 
AND transactions.Product_Category = visits_products_category.Product_Category
ORDER BY CA DESC
