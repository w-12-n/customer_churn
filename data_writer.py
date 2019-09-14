import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce
import numpy as np
import pandas as pd


class ChurnDataWriter(object):
    def __init__(self,
                 # if no orders after churn_thresh, then label customer as churned
                 churn_thresh=datetime.datetime(year=2018, month=8, day=1),
                 # time into future for prediction. e.g. model predicts customer will churn in 3 months
                 churn_in=relativedelta(months=3),
                 data_dir='./data/',
                 # order history filename
                 file='read_file.txt',
                 # column names in the read_file
                 # required
                 date='Date',
                 customer='Customer Name',
                 order_id='Order Number',
                 prod='Product Category',
                 sales='Sales',
                 costs='Cost',
                 # extra column names--if you don't have any, leave as empty lists
                 # extra features that are non-numeric
                 non_numeric=[],
                 # extra features that are averaged within an order
                 to_avg=[],
                 # extra features that are summed within an order
                 to_sum=[],
                 # how read_file separates its columns
                 sep='\t'):
        self.data_dir = data_dir

        self.to_avg = list(to_avg)
        self.to_sum = list(to_sum)
        self.numeric = [sales, costs] + self.to_avg + self.to_sum
        self.non_numeric = [date, order_id, prod, customer] + list(non_numeric)
        features = self.numeric + self.non_numeric

        self.df = self.read(file, features, sep)
        self.edit_df(prod, sales, costs, order_id, date, customer)
        self.x_flt, self.x_str, self.y = self.xy_from_df(churn_thresh, churn_in, date, customer)

    def read(self, file, features, sep):
        print('\nReading file... ')
        df = pd.read_csv(self.data_dir + file,
                         usecols=features,
                         sep=sep,
                         dtype={num: 'float32' for num in self.numeric},
                         thousands=',') \
            .dropna(axis=0, how='any')
        print('Success\n')
        return df

    def edit_df(self, prod, sales, costs, order_id, date, customer):
        self.categorize_sales(sales, costs, prod)
        self.unify_orders(order_id)
        self.get_interludes(customer, date)

    # Split sales and markup by product type
    def categorize_sales(self, sales, costs, prod):
        print('Categorizing sales... ')
        product_names = set(self.df[prod].values)
        for name in product_names:
            prod_sales, prod_markup = name+'_sales', name+'_markup'
            this_prod = self.df[prod] == name

            markup = self.df[sales] / np.maximum(self.df[costs], 1)
            self.df[prod_markup] = np.where(this_prod, markup, 0)
            self.df[prod_sales] = np.where(this_prod, self.df[sales], 0)

            self.to_sum += [prod_sales]
            self.to_avg += [prod_markup]
            self.numeric += [prod_sales, prod_markup]

        self.df.drop(columns=[prod, sales, costs], inplace=True)
        self.numeric.remove(sales)
        self.numeric.remove(costs)
        self.non_numeric.remove(prod)
        print('Success\n')

    # Combines order by summing/averaging numeric & carrying along non-numeric terms
    def unify_orders(self, order_id):
        print('Unifying orders... ')
        orders = self.df.groupby(order_id)

        df_sum = orders[self.to_sum].sum()
        duplicates = self.df.set_index(order_id).index.duplicated(keep='first')
        df_non_num = self.df.loc[~duplicates][self.non_numeric]
        self.non_numeric.remove(order_id)

        dfs = [df_sum, df_non_num]
        if self.to_avg:
            df_avg = orders[self.to_avg].mean()
            dfs.append(df_avg)

        merged = reduce(lambda left, right: pd.merge(left, right, on=order_id), dfs)
        self.df = merged
        print('Success\n')

    # Calculates days between orders
    def get_interludes(self, customer, date):
        print('Finding interludes between sales... ')
        waited = 'Waited'
        self.df[date] = pd.to_datetime(self.df[date])
        self.df.sort_values([customer, date], inplace=True)

        self.df[waited] = self.df[date].diff().dt.days
        mask = self.df[customer] != self.df[customer].shift(1)
        self.df[waited] = np.where(mask, -1, self.df[waited])

        self.numeric.append(waited)
        self.non_numeric.remove(date)
        self.non_numeric.remove(customer)
        print('Success\n')

    # Creates a numeric X, non numeric X, and y
    def xy_from_df(self, churn_def, churn_in, date, customer):
        print('Creating Xs and Y... ')
        x_flt, x_str, y = [], [], []
        for _, order_history in self.df.set_index(date).groupby(customer):
            churned = order_history.index[-1] <= churn_def
            if churned:
                last_date = order_history.index[-1]
                relev_history = order_history[:last_date - churn_in]
            else:
                relev_history = order_history[:churn_def - churn_in]

            if len(relev_history) > 1:
                x_flt.append(relev_history[self.numeric].to_numpy()[1:])
                x_str.append(relev_history[self.non_numeric].to_numpy()[1:])
                y.append(churned)
        print('Success\n')
        return np.array(x_flt), np.array(x_str), np.array(y)

    # Saves 3D numpy arrays to data file
    def write_xy(self):
        print(f'Saving Xs and Y... ')
        self.save('x_flt', self.x_flt)
        self.save('x_str', self.x_str)
        self.save('y', self.y)
        print('Success\n')
        print(f'Fraction churned = {sum(self.y) / len(self.y)}')
        print(f'# features = {self.x_flt[0].shape[-1] + self.x_str[0].shape[-1]}')
        print(f'# samples = {len(self.y)}')
        print('Complete.\n')

    def save(self, name, arr):
        np.save(f'{self.data_dir + name}.npy', arr, allow_pickle=True)


if __name__ == '__main__':
    writer = ChurnDataWriter()
    writer.write_xy()
