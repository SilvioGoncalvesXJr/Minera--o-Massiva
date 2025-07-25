import pandas as pd
import numpy as np

# 1. Importação dos dados
customers = pd.read_csv('olist_customers_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')

# 2. Número de pedidos por cliente
orders_per_customer = orders.groupby('customer_id').order_id.nunique().reset_index()
orders_per_customer.columns = ['customer_id', 'num_orders']

# 3. Valor total gasto por cliente
order_items['total_price'] = order_items['price'] + order_items['freight_value']
order_value = order_items.groupby('order_id')['total_price'].sum().reset_index()
orders_value = pd.merge(orders[['order_id', 'customer_id']], order_value, on='order_id', how='left')
total_spent = orders_value.groupby('customer_id')['total_price'].sum().reset_index()
total_spent.columns = ['customer_id', 'total_spent']

# 4. Ticket médio por cliente
avg_ticket = total_spent.merge(orders_per_customer, on='customer_id')
avg_ticket['avg_ticket'] = avg_ticket['total_spent'] / avg_ticket['num_orders']

# 5. Média das avaliações por cliente
orders_reviews = pd.merge(orders[['order_id', 'customer_id']], reviews[['order_id', 'review_score']], on='order_id', how='left')
avg_review = orders_reviews.groupby('customer_id')['review_score'].mean().reset_index()
avg_review.columns = ['customer_id', 'avg_review_score']

# 6. Frequência de compra (tempo médio entre pedidos)
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders_sorted = orders.sort_values(['customer_id', 'order_purchase_timestamp'])
orders_sorted['prev_order'] = orders_sorted.groupby('customer_id')['order_purchase_timestamp'].shift(1)
orders_sorted['days_between_orders'] = (orders_sorted['order_purchase_timestamp'] - orders_sorted['prev_order']).dt.days
avg_freq = orders_sorted.groupby('customer_id')['days_between_orders'].mean().reset_index()
avg_freq.columns = ['customer_id', 'avg_days_between_orders']

# 7. Unindo todas as features em um único dataframe
features = customers[['customer_id']].drop_duplicates()
features = features.merge(orders_per_customer, on='customer_id', how='left')
features = features.merge(total_spent, on='customer_id', how='left')
features = features.merge(avg_ticket[['customer_id', 'avg_ticket']], on='customer_id', how='left')
features = features.merge(avg_review, on='customer_id', how='left')
features = features.merge(avg_freq, on='customer_id', how='left')

# Visualizando as primeiras linhas das features finais
print(features.head())