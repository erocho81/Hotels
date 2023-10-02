--Union all years:
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020]


-- CTE with Union:
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT * FROM hotels

-- nights* ADR (daily rate for hotel), to check revenue by year and hotel:
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT 
arrival_date_year,
hotel,
round(sum((stays_in_weekend_nights+stays_in_week_nights)* ADR),2) as revenue
from hotels
group by arrival_date_year, hotel


-- Join the rest of the tables (althought the info was already there):
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT * FROM hotels
left join dbo.market_segment 
on hotels.market_segment= market_segment.market_segment
left join dbo.meal_cost
on hotels.meal = meal_cost.meal


-- Order Months by revenue
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT arrival_date_month,
round(sum((stays_in_weekend_nights+stays_in_week_nights)* ADR),2) as revenue

from hotels
group by arrival_date_month
order by revenue desc

--- Meal counts and percentages
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT 
meal,
COUNT (*) as meal_type_quantity,
round((COUNT(*) * 100.0) / (select COUNT(*) from hotels),2) AS Meal_Percentage

from hotels
group by meal


-- Order top 10 country by revenue. Total count and revenue per count.
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT top 10 country,
COUNT (*) as total_per_country,
round(sum((stays_in_weekend_nights+stays_in_week_nights)* ADR),2) as revenue,
round(sum((stays_in_weekend_nights+stays_in_week_nights)* ADR)/ (COUNT(*)),2) AS Revenue_per_count

from hotels
group by country
order by revenue desc

-- customer_type information
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT customer_type,
SUM(stays_in_week_nights+stays_in_weekend_nights) as total_nights,
round(sum((stays_in_weekend_nights+stays_in_week_nights)* ADR),2) as revenue,
round((SUM(stays_in_weekend_nights+stays_in_week_nights) * 100 )/SUM(SUM(stays_in_weekend_nights+stays_in_week_nights)) OVER (),2) as total_nights_percentage

from hotels
group by customer_type
ORDER BY total_nights desc

-- How many room types have been changed?
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT
count (*) as changed_room_type
from hotels
where reserved_room_type != assigned_room_type

-- Reservation_Status
WITH hotels as (
SELECT * FROM dbo.[2018]
UNION
SELECT * FROM dbo.[2019]
UNION
SELECT * FROM dbo.[2020])

SELECT
reservation_status,
count (*) as qty_reservation_status
from hotels

group by reservation_status



--
