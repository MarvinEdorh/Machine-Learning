WITH visitors AS (
SELECT fullvisitorid,device.deviceCategory,device.operatingSystem,trafficSource.campaign, trafficSource.medium, geoNetwork.continent,
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201701*` AS ga, UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp ), 
visits_products AS (  
SELECT fullvisitorid, hp.v2ProductName AS Products, SUM( totals.visits) AS Products_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201701*` AS ga,UNNEST(ga.hits) AS hits, 
UNNEST(hits.product) AS hp 
GROUP BY fullvisitorid, Products ORDER BY fullvisitorid, Products_Visits DESC ), 
visits_products_category AS (  
SELECT fullvisitorid, hp.v2ProductCategory AS Products_Category, SUM( totals.visits) Products_Category_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201701*` AS ga,UNNEST(ga.hits) AS hits, 
UNNEST(hits.product) AS hp 
GROUP BY fullvisitorid, Products_Category ORDER BY fullvisitorid, Products_Category_Visits DESC )
SELECT visitors.fullvisitorid, deviceCategory, operatingSystem, campaign, medium, continent, 
Products, Products_Category,Products_Visits, Products_Category_Visits, FROM visitors
LEFT JOIN visits_products
USING (fullvisitorid)
LEFT JOIN visits_products_category
USING (fullvisitorid)