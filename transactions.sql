WITH 
transactions AS (
SELECT hits.transaction.transactionId AS ID_Transaction, device.deviceCategory,device.operatingSystem,
trafficSource.campaign, trafficSource.medium, geoNetwork.continent, fullvisitorid , hp.v2ProductName AS Products, 
hp.v2ProductCategory AS Products_Category,IFNULL(SUM(hits.transaction.transactionRevenue/1000000),0) AS CA, 
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20161201*` AS ga, UNNEST(ga.hits) AS hits,UNNEST(hits.product) as hp 
GROUP BY ID_Transaction, device.deviceCategory,device.operatingSystem,trafficSource.campaign, trafficSource.medium, geoNetwork.continent,
fullvisitorid , Products, Products_Category
ORDER BY CA DESC,ID_Transaction ), 
visits_products AS (  
SELECT fullvisitorid, hp.v2ProductName AS Products, SUM( totals.visits) AS Products_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,UNNEST(ga.hits) AS hits,UNNEST(hits.product) as hp
WHERE _TABLE_SUFFIX <= '20161201'
GROUP BY fullvisitorid, Products ORDER BY fullvisitorid, Products_Visits DESC ), 
visits_products_category AS (  
SELECT fullvisitorid, hp.v2ProductCategory AS Products_Category, SUM( totals.visits) Products_Category_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,UNNEST(ga.hits) AS hits,UNNEST(hits.product) as hp
WHERE _TABLE_SUFFIX <= '20161201'
GROUP BY fullvisitorid, Products_Category ORDER BY fullvisitorid, Products_Category_Visits DESC )
SELECT ID_Transaction, deviceCategory, operatingSystem, campaign, medium, continent, transactions.fullvisitorid,
transactions.Products, Products_Visits, transactions.Products_Category, Products_Category_Visits, CA FROM transactions
LEFT JOIN visits_products
ON transactions.fullvisitorid = visits_products.fullvisitorid AND transactions.Products = visits_products.Products
LEFT JOIN visits_products_category
ON transactions.fullvisitorid = visits_products_category.fullvisitorid AND transactions.Products_Category = visits_products_category.Products_Category
ORDER BY CA DESC