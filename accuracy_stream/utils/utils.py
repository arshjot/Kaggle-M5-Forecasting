import numpy as np
from itertools import product


def get_aggregated_series(sales, item_id, dept_id, cat_id, store_id, state_id):
    """
    Aggregates 30,490 level 12 series to generate data for all 42,840 series

    Input data format:
    sales: np array of shape (30490, num_timesteps)
    all id arguments: np arrays of shape (30490,)
    """

    aggregated_series, aggregated_series_id = np.empty((42840, sales.shape[1])), np.empty(42840, '<U28')

    # Level 1
    aggregated_series[0] = sales.sum(0)
    aggregated_series_id[0] = 'Level1_Total_X'
    fill_id = 1

    # Level 2
    for agg_element in np.unique(state_id):
        agg_sales = sales[np.where(state_id == agg_element)[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level2_{agg_element}_X'
        fill_id += 1

    # Level 3
    for agg_element in np.unique(store_id):
        agg_sales = sales[np.where(store_id == agg_element)[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level3_{agg_element}_X'
        fill_id += 1

    # Level 4
    for agg_element in np.unique(cat_id):
        agg_sales = sales[np.where(cat_id == agg_element)[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level4_{agg_element}_X'
        fill_id += 1

    # Level 5
    for agg_element in np.unique(dept_id):
        agg_sales = sales[np.where(dept_id == agg_element)[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level5_{agg_element}_X'
        fill_id += 1

    # Level 6
    for agg_1, agg_2 in product(np.unique(state_id), np.unique(cat_id)):
        agg_sales = sales[np.where((state_id == agg_1) & (cat_id == agg_2))[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level6_{agg_1}_{agg_2}'
        fill_id += 1

    # Level 7
    for agg_1, agg_2 in product(np.unique(state_id), np.unique(dept_id)):
        agg_sales = sales[np.where((state_id == agg_1) & (dept_id == agg_2))[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level7_{agg_1}_{agg_2}'
        fill_id += 1

    # Level 8
    for agg_1, agg_2 in product(np.unique(store_id), np.unique(cat_id)):
        agg_sales = sales[np.where((store_id == agg_1) & (cat_id == agg_2))[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level8_{agg_1}_{agg_2}'
        fill_id += 1

    # Level 9
    for agg_1, agg_2 in product(np.unique(store_id), np.unique(dept_id)):
        agg_sales = sales[np.where((store_id == agg_1) & (dept_id == agg_2))[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level9_{agg_1}_{agg_2}'
        fill_id += 1

    # Level 10
    for agg_element in np.unique(item_id):
        agg_sales = sales[np.where(item_id == agg_element)[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level10_{agg_element}_X'
        fill_id += 1

    # Level 11
    for agg_1, agg_2 in product(np.unique(state_id), np.unique(item_id)):
        agg_sales = sales[np.where((state_id == agg_1) & (item_id == agg_2))[0]].sum(0)[np.newaxis, :]
        aggregated_series[fill_id] = agg_sales
        aggregated_series_id[fill_id] = f'Level11_{agg_1}_{agg_2}'
        fill_id += 1

    # Level 12
    aggregated_series[fill_id:] = sales
    aggregated_series_id[fill_id:] = np.array([f'Level12_{item}_{store}'
                                               for item, store in zip(item_id, store_id)])

    # Return the arrays sorted acc to ids
    sort_idx = aggregated_series_id.argsort()

    return aggregated_series[sort_idx], aggregated_series_id[sort_idx]


def get_weights_all_levels(sales, sell_price, item_id, dept_id, cat_id, store_id, state_id):
    """
    Generates weights for all 42,840 series

    Input data format:
    sales: np array of shape (30490, 28)
    sell_price: np array of shape (30490, 28)

    all id arguments: np arrays of shape (30490,)
    """

    assert (sales.shape == sell_price.shape), "Sell price and Sales arrays have different sizes"
    assert (sales.shape[1] == 28), "Number of timesteps provided weight calculation is not equal to 28"

    # Get actual dollar sales for last 28 days for all 42,840 series
    dollar_sales = sales * sell_price
    agg_series, agg_series_id = get_aggregated_series(dollar_sales, item_id, dept_id, cat_id, store_id, state_id)

    # Sum up the actual dollar sales for all 28 timesteps
    agg_series = agg_series.sum(1)

    # Calculate total sales for each level
    level_totals = agg_series[np.core.defchararray.find(agg_series_id, f'Level1_') == 0].sum()

    # Calculate weight for each series
    weights = agg_series / level_totals

    return weights, agg_series_id
