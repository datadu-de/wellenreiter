import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt


def create_df_from_csv(
    filename="data-12m-rl.txt", whereto_program=None, whereto_sales=None
):

    df = pd.read_csv(
        filename,
        ";",
        parse_dates=["date"],
        infer_datetime_format=True,
        dtype={
            "measure": np.str_,
            "value": np.int_,
        },
    )

    initial_program = df[df["measure"] == "program"].sum()["value"]
    initial_sales = df[df["measure"] == "sales"].sum()["value"]

    df = df.pivot_table(columns="measure", values="value", index="date")

    whereto_program.empty()
    whereto_sales.empty()

    whereto_program.write(
        """
        ## program
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
    program = []
    sales = []
    for d in df.index.to_series():
        # ref_date = df.index.to_series().dt.date[d]
        label = d.strftime("%B %Y")
        key = f"{keypart}_{d.strftime('%Y-%m-%d')}"
        program.append(
            whereto_program.number_input(
                label,
                key=f"program_{key}",
                value=df.at[d, "program"],
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
    return df, initial_program, initial_sales, program, sales


def write_matrix(initial_program, initial_sales, df, whereto_matrix=None):

    if whereto_matrix is None:
        whereto_matrix = st.sidebar

    current_program = df["program"].sum()
    current_sales = df["sales"].sum()
    table_md = f"""
    |measure|initial value|current value| diff |
    |--|--|--|--|
    |program|{"{:,}".format(initial_program)}|{"{:,.0f}".format(current_program)}|{"{:+,.0f}".format(current_program-initial_program)}|
    |sales|{"{:,}".format(initial_sales)}|{"{:,.0f}".format(current_sales)}|{"{:+,.0f}".format(current_sales-initial_sales)}|
    """
    whereto_matrix.markdown(table_md)


def init_params(whereto_params=None):
    if whereto_params is None:
        whereto_params = st.sidebar

    whereto_params.write("## parameters")

    program_boost = whereto_params.slider(
        "program boost [%]",
        50,
        150,
        100,
        1,
        help="multiply each program value by this boost factor in % (default: 100%)",
    )
    sales_boost = whereto_params.slider(
        "sales boost [%]",
        50,
        150,
        100,
        1,
        help="multiply each sales value by this boost factor in % (default: 100%)",
    )
    program_base = whereto_params.number_input(
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
        value=150_000,
        step=1,
        format="%i",
        help="min critical stock level marked in the chart",
    )

    critical_stock_max = whereto_params.number_input(
        "max critical stock level",
        # min_value=-10,
        # max_value=10,
        value=200_000,
        step=1,
        format="%i",
        help="max critical stock level marked in the chart",
    )

    target_value = whereto_params.number_input(
        "YE target",
        # min_value=-10,
        # max_value=10,
        value=170_000,
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
        program_boost,
        sales_boost,
        program_base,
        critical_stock_min,
        critical_stock_max,
        target_value,
        uploaded_file,
    )


def init_layout():
    err = st.empty()
    c1, c2, c3, c4 = st.beta_columns((3, 1, 1, 1))

    c1.write(
        """
        # wellenreiter revamped
        
        made with :brain: by [Daimler TSS CloudAnalytics](https://social.intra.corpintra.net/groups/cloud-analytics)
        """
    )

    return (c1, c2, c3, c4, err)


def main(*args, **kwargs):

    st.set_page_config(
        page_title="wellenreiter | by CloudAnalytics @DaimlerTSS ", layout="wide"
    )

    c1, c2, c3, c4, err = init_layout()

    (
        program_boost,
        sales_boost,
        program_base,
        critical_stock_min,
        critical_stock_max,
        target_value,
        uploaded_file,
    ) = init_params(whereto_params=c4)

    if uploaded_file is None:
        df, initial_program, initial_sales, program, sales = create_df_from_csv(
            whereto_program=c2, whereto_sales=c3
        )
    else:
        try:
            df, initial_program, initial_sales, program, sales = create_df_from_csv(
                uploaded_file, whereto_program=c2, whereto_sales=c3
            )

        except ValueError:
            err.error(
                """
                The csv file do not have a valid format. Please upload a csv file with the columns
                ```
                date;       measure;    value
                2021-01-01; program;    20
                2021-01-01; sales;      20
                ```
                and semicolon as separator.
                """,
            )
            st.stop()

    # override values from csv with widget values
    df["program"] = program
    df["sales"] = sales

    # apply boost
    df["program"] *= program_boost / 100
    df["sales"] *= sales_boost / 100

    # round up
    df["program"] = df["program"].apply(np.ceil)
    df["sales"] = df["sales"].apply(np.ceil)

    # calculate stock and add to df
    df["stock"] = df["program"].mask(pd.isnull, 0) - df["sales"].mask(pd.isnull, 0)
    df["stock"].iloc[0] += program_base

    # add running totals
    df["program_rt"] = df["program"].cumsum(axis=0)
    df["sales_rt"] = df["sales"].cumsum(axis=0)
    df["stock_rt"] = df["stock"].cumsum(axis=0)
    df["stock"].iloc[0] -= program_base

    df = df.reset_index()
    df["mmm"] = df["date"].dt.strftime("%b")

    # refresh overview table
    domain = ["program", "sales", "stock"]
    range_ = ["#0082b3", "#a8e9b3", "#febf00"]  # blue, green, orange

    fig1, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    ax1.set_frame_on(False)
    ax1.set_axis_off()
    ax2.set_frame_on(False)
    ax2.set_axis_on()

    ax2.fill_between(
        df["mmm"],
        critical_stock_min,
        critical_stock_max,
        label="critial stock level",
        alpha=0.15,
    )

    ax2.fill_between(
        df["mmm"],
        df["program"],
        df["sales"],
        df["program"] > df["sales"],
        interpolate=True,
        alpha=0.15,
        color="red",
        label="stock increase",
    )
    ax2.fill_between(
        df["mmm"],
        df["program"],
        df["sales"],
        df["program"] < df["sales"],
        interpolate=True,
        alpha=0.15,
        color="green",
        label="stock decrease",
    )
    ax2.plot(df["mmm"], df["program"], label="program", color="#0082b3")
    ax2.plot(df["mmm"], df["sales"], label="sales", color="#a8e9b3")
    ax2.plot(df["mmm"], df["stock_rt"], label="stock", color="#febf00")
    ax2.plot(df["mmm"].iat[-1], target_value, "bo", label="YE target")

    # ax1.yaxis.set_major_formatter(FormatStrFormatter("%.f"))
    ax2.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, pos: (f"{x:,.0f}"))
    )
    ax1.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, pos: (f"{x:,.0f}"))
    )

    df["stock_decrease"] = df.apply(lambda x: x["program"] > x["sales"], axis=1)
    df["stock_increase"] = df.apply(lambda x: x["program"] < x["sales"], axis=1)
    df["stock_color"] = df.apply(
        lambda x: x["program"] > x["sales"] and "red" or "green", axis=1
    )

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

    # ax1.legend()
    # plt.style.use("default")
    # plt.style.use("seaborn")
    # plt.style.use("seaborn-poster")
    # plt.style.use("seaborn-whitegrid")
    plt.style.use("tableau-colorblind10")
    # mpl.rc("axes", grid=True, edgecolor="white")
    # mpl.rc("grid", alpha=0.25)
    # mpl.rc("bar", width=0.5)
    plt.tight_layout()

    c1.write(
        """
        ## program, sales & stock
        ![#0082b3](https://via.placeholder.com/15/0082b3/000000?text=+) program 
        ![#a8e9b3](https://via.placeholder.com/15/a8e9b3/000000?text=+) sales 
        ![#febf00](https://via.placeholder.com/15/febf00/000000?text=+) stock 
        ![#ff6666](https://via.placeholder.com/15/ff6666/000000?text=+) stock increase 
        ![#66b266](https://via.placeholder.com/15/66b266/000000?text=+) stock decrease
        """
    )
    c1.pyplot(fig1)

    write_matrix(initial_program, initial_sales, df, whereto_matrix=c1)

    with c1.beta_expander("input data", expanded=False):
        st.dataframe(df.set_index("date"))


if __name__ == "__main__":
    main()
