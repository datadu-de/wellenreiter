import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt


def create_df_from_csv(filename="data-12m.csv", whereto_production=None, whereto_sales=None):
    df = pd.read_csv(
        filename,
        sep=",",
        parse_dates=["date"],
        date_format="%Y-%m-%d",
        dtype={
            "measure": np.str_,
            "value": np.int_,
        },
    )

    initial_production = df[df["measure"] == "production"]["value"].sum()
    initial_sales = df[df["measure"] == "sales"]["value"].sum()

    df = df.pivot_table(columns="measure", values="value", index="date")

    whereto_production.empty()
    whereto_sales.empty()

    whereto_production.write(
        """
        ## production
        """
    )
    whereto_sales.write(
        """
        ## sales
        """
    )
    _kw = {
        "step": 100,
        "format": "%d",
        # "min_value": 0,
        # "max_value": 1_000_000,
        "help": "increments of 100",
    }

    keypart = type(filename) is str and filename or filename.name
    production = []
    sales = []
    for d in df.index.to_series():
        # ref_date = df.index.to_series().dt.date[d]
        label = d.strftime("%B %Y")
        key = f"{keypart}_{d.strftime('%Y-%m-%d')}"
        production.append(
            whereto_production.number_input(
                label,
                key=f"production_{key}",
                value=df.at[d, "production"],
                **_kw,
            )
        )
        sales.append(
            whereto_sales.number_input(
                label,
                key=f"sales_{key}",
                value=df.at[d, "sales"],
                **_kw,
            )
        )
    return df, initial_production, initial_sales, production, sales


def write_matrix(initial_production, initial_sales, df, whereto_matrix=None):
    if whereto_matrix is None:
        whereto_matrix = st.sidebar

    initial_stock = df["stock_rt"].iat[0]

    current_production = df["production"].sum()
    current_sales = df["sales"].sum()
    current_stock = df["stock_rt"].iat[-1]

    whereto_matrix.table(
        pd.DataFrame(
            {
                # "measure": ["production", "sales", "stock"],
                "initial value": [initial_production, initial_sales, initial_stock],
                "current value": [current_production, current_sales, current_stock],
                "diff": [
                    current_production - initial_production,
                    current_sales - initial_sales,
                    current_stock - initial_stock,
                ],
            },
            index=["production", "sales", "stock"],
        )
    )


def init_params(whereto_params=None):
    if whereto_params is None:
        whereto_params = st.sidebar

    whereto_params.write("## parameters")

    production_boost = whereto_params.slider(
        "production boost [%]",
        50,
        150,
        100,
        1,
        help="multiply each production value by this boost factor in % (default: 100%)",
    )
    sales_boost = whereto_params.slider(
        "sales boost [%]",
        50,
        150,
        100,
        1,
        help="multiply each sales value by this boost factor in % (default: 100%)",
    )
    production_base = whereto_params.number_input(
        "initial stock value",
        # min_value=-500000,
        # max_value=500000,
        value=80_000,
        step=1,
        format="%i",
        help="value is added as baseline",
    )
    critical_stock_min = whereto_params.number_input(
        "min critical stock level",
        # min_value=-10,
        # max_value=10,
        value=55_000,
        step=1,
        format="%i",
        help="min critical stock level marked in the chart",
    )

    critical_stock_max = whereto_params.number_input(
        "max critical stock level",
        # min_value=-10,
        # max_value=10,
        value=80_000,
        step=1,
        format="%i",
        help="max critical stock level marked in the chart",
    )

    target_value = whereto_params.number_input(
        "YE target",
        # min_value=-10,
        # max_value=10,
        value=65_000,
        step=1,
        format="%i",
        help="target value at the end of the input window",
    )
    uploaded_file = whereto_params.file_uploader(
        label="Choose a csv file",
        accept_multiple_files=False,
        help="Only the standard csv format is valid.",
    )

    return (
        production_boost,
        sales_boost,
        production_base,
        critical_stock_min,
        critical_stock_max,
        target_value,
        uploaded_file,
    )


def init_layout():
    err = st.empty()
    c1, c2, c3, c4 = st.columns((3, 1, 1, 1))

    c1.write(
        """
        # wellenreiter
        
        with :brain: by [Frank](https://github.com/users/datadu-de),
        code at [github](https://github.com/datadu-de/wellenreiter)
        """
    )

    return (c1, c2, c3, c4, err)


def main(*args, **kwargs):
    st.set_page_config(page_title="wellenreiter | by Frank", layout="wide")

    c1, c2, c3, c4, err = init_layout()

    (
        production_boost,
        sales_boost,
        production_base,
        critical_stock_min,
        critical_stock_max,
        target_value,
        uploaded_file,
    ) = init_params(whereto_params=c4)

    if uploaded_file is None:
        df, initial_production, initial_sales, production, sales = create_df_from_csv(
            whereto_production=c2, whereto_sales=c3
        )
    else:
        try:
            (
                df,
                initial_production,
                initial_sales,
                production,
                sales,
            ) = create_df_from_csv(uploaded_file, whereto_production=c2, whereto_sales=c3)

        except ValueError:
            err.error(
                """
                The csv file do not have a valid format. Please upload a csv file with the columns
                ```
                date;       measure;    value
                2021-01-01; production;    20
                2021-01-01; sales;      20
                ```
                and semicolon as separator.
                """,
            )
            st.stop()

    # override values from csv with widget values
    df["production"] = production
    df["sales"] = sales

    # apply boost
    df["production"] *= production_boost / 100
    df["sales"] *= sales_boost / 100

    # round up
    df["production"] = df["production"].apply(np.ceil)
    df["sales"] = df["sales"].apply(np.ceil)

    # calculate stock and add to df
    df["stock"] = df["production"].mask(pd.isnull, 0) - df["sales"].mask(pd.isnull, 0)
    df["stock"].iloc[0] += production_base

    # add running totals
    df["production_rt"] = df["production"].cumsum(axis=0)
    df["sales_rt"] = df["sales"].cumsum(axis=0)
    df["stock_rt"] = df["stock"].cumsum(axis=0)
    df["stock"].iloc[0] -= production_base

    df = df.reset_index()
    df["mmm"] = df["date"].dt.strftime("%b")

    # plt.style.use("default")
    # plt.style.use("tableau-colorblind10")

    plt.style.use("seaborn")

    fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 3]})

    ax1.set_frame_on(False)
    ax1.set_axis_off()
    ax2.set_frame_on(False)
    ax2.set_axis_on()

    ax2.fill_between(
        df["mmm"],
        critical_stock_min,
        critical_stock_max,
        label="critial stock level",
        alpha=0.1,
    )

    ax2.fill_between(
        df["mmm"],
        df["production"],
        df["sales"],
        df["production"] > df["sales"],
        interpolate=True,
        alpha=0.15,
        color="red",
        label="stock increase",
    )
    ax2.fill_between(
        df["mmm"],
        df["production"],
        df["sales"],
        df["production"] < df["sales"],
        interpolate=True,
        alpha=0.15,
        color="green",
        label="stock decrease",
    )
    ax2.plot(df["mmm"], df["production"], label="production")
    ax2.plot(df["mmm"], df["sales"], label="sales")
    ax2.plot(df["mmm"], df["stock_rt"], label="stock")
    ax2.plot(df["mmm"].iat[-1], target_value, "bo", label="YE target")

    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: (f"{x:,.0f}")))

    ax2.legend()

    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: (f"{x:,.0f}")))

    df["stock_decrease"] = df.apply(lambda x: x["production"] > x["sales"], axis=1)
    df["stock_increase"] = df.apply(lambda x: x["production"] < x["sales"], axis=1)
    df["stock_color"] = df.apply(lambda x: x["production"] > x["sales"] and "red" or "green", axis=1)

    ax1.bar(
        df["mmm"],
        df["stock"],
        0.5,
        color=df["stock_color"],
        alpha=0.6,
        edgecolor=df["stock_color"],
        linewidth=2,
    )

    for i in df.index:
        ax1.annotate(
            f"{df['stock'].iat[i]:,.0f}",
            (df["mmm"].iat[i], df["stock"].iat[i]),
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to points (x,y)
            ha="center",
        )

    plt.tight_layout()

    # c1.write(
    #     """
    #     ## production, sales & stock
    #     ![#0082b3](https://via.placeholder.com/15/0082b3/000000?text=+) production
    #     ![#a8e9b3](https://via.placeholder.com/15/a8e9b3/000000?text=+) sales
    #     ![#febf00](https://via.placeholder.com/15/febf00/000000?text=+) stock
    #     ![#ff6666](https://via.placeholder.com/15/ff6666/000000?text=+) stock increase
    #     ![#66b266](https://via.placeholder.com/15/66b266/000000?text=+) stock decrease
    #     """
    # )
    c1.markdown("## production, sales & stock")
    c1.pyplot(fig1)

    write_matrix(initial_production, initial_sales, df, whereto_matrix=c1)

    with c1.expander("input data", expanded=False):
        st.dataframe(df.set_index("date"))


if __name__ == "__main__":
    main()
