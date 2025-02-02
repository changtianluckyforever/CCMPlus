import time

from datetime import datetime, timedelta

from pytrends.request import TrendReq
import pandas as pd

trend = TrendReq(hl='en-US', tz=360)


def get_historical_interest(keywords, year_start=2018, month_start=1,
                            day_start=1, hour_start=0, year_end=2018,
                            month_end=2, day_end=1, hour_end=0, cat=0,
                            geo='', gprop='', sleep=0, frequency='hourly'):
    """Gets historical hourly data for interest by chunking requests to 1 week at a time (which is what Google allows)"""

    # construct datetime objects - raises ValueError if invalid parameters
    initial_start_date = start_date = datetime(year_start, month_start,
                                               day_start, hour_start)
    end_date = datetime(year_end, month_end, day_end, hour_end)

    # Timedeltas:
    # 7 days for hourly
    # ~250 days for daily (270 seems to be max but sometimes breaks?)
    # For weekly can pull any date range so no method required here

    delta = timedelta(days=7)

    df = pd.DataFrame()

    date_iterator = start_date
    date_iterator += delta

    while True:
        # format date to comply with API call (different for hourly/daily)
        start_date_str = start_date.strftime('%Y-%m-%dT%H')
        date_iterator_str = date_iterator.strftime('%Y-%m-%dT%H')

        tf = start_date_str + ' ' + date_iterator_str
        print(tf)

        try:
            trend.build_payload(keywords, cat, tf, geo, gprop)
            week_df = trend.interest_over_time()
            df = pd.concat([df, week_df], ignore_index=True)
        except Exception as e:
            print(e)
            pass

        start_date += delta
        date_iterator += delta

        if date_iterator > end_date:
            # Run more days to get remaining data that would have been truncated if we stopped now
            start_date_str = start_date.strftime('%Y-%m-%dT%H')
            date_iterator_str = date_iterator.strftime('%Y-%m-%dT%H')

            tf = start_date_str + ' ' + date_iterator_str

            try:
                trend.build_payload(keywords, cat, tf, geo, gprop)
                week_df = trend.interest_over_time()
                df = pd.concat([df, week_df], ignore_index=True)
            except Exception as e:
                print(e)
                pass
            break

        # just in case you are rate-limited by Google. Recommended is 60 if you are.
        if sleep > 0:
            time.sleep(sleep)

    # Return the dataframe with results from our timeframe
    return df.loc[initial_start_date:end_date]


datat = {}
i = 0
kw_list = ['translate', 'food delivery', 'shared bicycle', 'urber', 'image classification']

for kw in kw_list:
    get_historical_interest(kw, year_start=2023, month_start=2, day_start=1, hour_start=0, year_end=2023,
                            month_end=2, day_end=2, hour_end=23, cat=0, geo='US', gprop='', sleep=5)
    datat[i] = trend.interest_over_time()
    i += 1

trendframe = pd.concat(datat, axis=1)
trendframe.columns = trendframe.columns.droplevel(0)
trendframe = trendframe.drop('isPartial', axis=1)

trendframe.to_csv("trendframe.csv")
