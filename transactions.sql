#On selectionne l'id, l'utilisateur, la date, l'heure, le device, l'os, la soure, la campagne, le pays, 
#les categories produits et produits achetés pour chaque transaction, 1 si la transaction a eu lieu 0 sinon.
WITH transactions AS (
SELECT DISTINCT hits.transaction.transactionId, fullvisitorid,
       DATETIME(EXTRACT(YEAR FROM PARSE_DATE("%Y%m%d", date)), EXTRACT(MONTH FROM PARSE_DATE("%Y%m%d", date)),
                EXTRACT(DAY FROM PARSE_DATE("%Y%m%d", date)), hits.hour, hits.minute, 00) AS datetime_transaction,
       device.deviceCategory, device.operatingSystem,trafficSource.campaign, trafficSource.medium, geoNetwork.country, 
       hp.v2ProductCategory, hp.v2ProductName, hp.productPrice /1000000 AS price,
       CASE WHEN hits.transaction.transactionId IS NULL THEN 0 ELSE 1 END AS transaction
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp),

#On calcule le nombre de visites de chaque utlisateur sur chaque produit en fonction du temps.
product AS (
SELECT fullvisitorid, hp.v2ProductName, 
       DATETIME(EXTRACT(YEAR FROM PARSE_DATE("%Y%m%d", date)), EXTRACT(MONTH FROM PARSE_DATE("%Y%m%d", date)),
                EXTRACT(DAY FROM PARSE_DATE("%Y%m%d", date)), hits.hour, hits.minute, 00) AS datetime, 
       SUM(totals.visits) AS visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp 
GROUP BY fullvisitorid, hp.v2ProductName, datetime),

#On additionne le nombre de visites 1 à 1 par utlisateur et par produit dans l'ordre croissant du temps
product_visits AS (
SELECT fullvisitorid, v2ProductName, datetime, 
       SUM(visits) OVER(PARTITION BY fullvisitorid, v2ProductName ORDER BY datetime 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS nb_product_visits FROM product),

#On calcule le nombre de visites de chaque utlisateur sur chaque categorie produit en fonction du temps.
category AS (
SELECT fullvisitorid, hp.v2ProductCategory,
       DATETIME(EXTRACT(YEAR FROM PARSE_DATE("%Y%m%d", date)), EXTRACT(MONTH FROM PARSE_DATE("%Y%m%d", date)),
                EXTRACT(DAY FROM PARSE_DATE("%Y%m%d", date)), hits.hour, hits.minute, 00) AS datetime, 
       SUM(totals.visits) AS visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp 
GROUP BY fullvisitorid, hp.v2ProductCategory, datetime),

#On additionne le nombre de visites 1 à 1 par utlisateur et par categorie produit dans l'ordre croissant du temps
category_visits AS (
SELECT fullvisitorid, v2ProductCategory, datetime, 
       SUM(visits) OVER(PARTITION BY fullvisitorid, v2ProductCategory ORDER BY datetime 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS nb_category_visits FROM category)

#On selectionne l'id, le device, l'os, la source, la campagne, le pays, les categories produits et produits achetés
#pour chaque transaction ainsi que le nombre de visites que l'utlisateur a fait sur les categories produits et 
#produits consommés avant qu'il effectue la transaction, 1 si la transaction a eu lieu 0 sinon.
SELECT transactionId, deviceCategory, operatingSystem, campaign, medium, country, transactions.v2ProductCategory, 
       transactions.v2ProductName, price, nb_product_visits, nb_category_visits, transaction FROM transactions 
LEFT JOIN product_visits 
ON transactions.fullvisitorid = product_visits.fullvisitorid 
AND transactions.v2ProductName = product_visits.v2ProductName 
AND transactions.datetime_transaction = product_visits.datetime
LEFT JOIN category_visits 
ON transactions.fullvisitorid = category_visits.fullvisitorid 
AND transactions.v2ProductCategory = category_visits.v2ProductCategory 
AND transactions.datetime_transaction = category_visits.datetime
ORDER BY transaction DESC, Datetime_transaction 

################################################### prediction ##########################################################

SELECT DISTINCT fullvisitorid, device.deviceCategory, device.operatingSystem,trafficSource.campaign, 
trafficSource.medium, geoNetwork.country, hp.v2ProductCategory, hp.v2ProductName, hp.productPrice /1000000 AS price,
SUM(totals.visits) OVER(PARTITION BY fullvisitorid, v2ProductName) AS product_visits,
SUM(totals.visits) OVER(PARTITION BY fullvisitorid, v2ProductCategory) AS category_visits,  
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp WHERE _TABLE_SUFFIX <= '20161201'
ORDER BY product_visits DESC
