import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime

# STEP 1: PREPROCESS DATA
def preprocess_data(base_dir="./data/data"):
    data_dict = {}

    for folder in ["PR", "GHI"]:
        folder_path = os.path.join(base_dir, folder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    try:
                        file_name = file.replace(".csv", "")
                        date = datetime.strptime(file_name, "%Y-%m-%d").date()

                        df_csv = pd.read_csv(full_path)
                        value = pd.to_numeric(df_csv.select_dtypes(include=['number']).values.flatten(), errors='coerce')
                        value = value[~pd.isna(value)]
                        value = value[0] if len(value) > 0 else None

                        if folder == "PR":
                            data_dict.setdefault(date, {})["PR"] = value
                        elif folder == "GHI":
                            data_dict.setdefault(date, {})["GHI"] = value

                    except Exception as e:
                        print(f"⚠️ Skipping file {file}: {e}")

    rows = []
    for date, values in data_dict.items():
        rows.append({"Date": date, "GHI": values.get("GHI"), "PR": values.get("PR")})

    if not rows:
        print("❌ No data found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Date")
    df["PR"] = pd.to_numeric(df["PR"], errors="coerce")
    df["GHI"] = pd.to_numeric(df["GHI"], errors="coerce")

    df.to_csv("combined_data.csv", index=False)
    print(f"✅ Saved combined_data.csv with {len(df)} rows")
    return df

# STEP 2: PLOT GRAPH
def plot_pr_graph(df):
    if df.empty:
        print("❌ No data to plot.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['PR_MA_30'] = df['PR'].rolling(window=30).mean()

    def get_budget(date):
        base = 73.9
        start = pd.Timestamp("2019-07-01")
        years = max(0, (date - start).days // 365)
        return base * ((1 - 0.008) ** years)

    df['Budget_PR'] = df['Date'].apply(get_budget)

    def get_color(ghi):
        if pd.isna(ghi): return 'gray'
        elif ghi < 2: return 'navy'
        elif ghi < 4: return 'lightblue'
        elif ghi < 6: return 'orange'
        else: return 'brown'

    df['Color'] = df['GHI'].apply(get_color)

    valid_df = df[df['PR'].notna()]
    above_count = (valid_df['PR'] > valid_df['Budget_PR']).sum()
    total_count = len(valid_df)

    #  SQUARE FIGURE WITH MARGINS
    fig = plt.figure(figsize=(13, 13))
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95)

    # SCATTER + LINES
    plt.scatter(df['Date'], df['PR'], c=df['Color'], s=20, alpha=0.8, label='Daily PR values (colored by GHI)')
    plt.plot(df['Date'], df['PR_MA_30'], color='red', linewidth=2, label="30–d moving average of PR")
    plt.plot(df['Date'], df['Budget_PR'], color='darkgreen', linewidth=2, label="Target Budget Yield Performance Ratio")

    # TITLE AND AXES
    plt.title("Performance Ratio Evolution\nFrom {} to {}".format(df['Date'].min().date(), df['Date'].max().date()), fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Performance Ratio [%]", fontsize=12)
    plt.grid(True)

    # COLOR LEGEND FOR GHI RANGES
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='GHI < 2', markerfacecolor='navy', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='GHI 2–4', markerfacecolor='lightblue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='GHI 4–6', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='GHI > 6', markerfacecolor='brown', markersize=8),
        Line2D([0], [0], color='red', label='30–d moving average of PR'),
        Line2D([0], [0], color='darkgreen', label='Target Budget Yield Performance Ratio')
    ]
    plt.legend(handles=legend_elements, title="Graph Labels", loc='upper left')

    # BUDGET LINE TEXT LABEL
    y1 = 73.9
    y2 = round(y1 * (1 - 0.008), 1)
    y3 = round(y1 * (1 - 0.008)**2, 1)
    budget_label = f"Budget PR: [1Y={y1}%, 2Y={y2}%, 3Y={y3}%]"
    plt.text(df['Date'].min(), 75.5, budget_label, fontsize=10, color='darkgreen')

    # ABOVE BUDGET POINTS COUNT
    if total_count > 0:
        annotation = f"Points above Target Budget PR = {above_count}/{total_count} = {above_count/total_count:.1%}"
    else:
        annotation = "Points above Target Budget PR = N/A"
    plt.text(df['Date'].min(), 73.5, annotation, fontsize=10, color='black')

    # SUMMARY BOX
    today = df['Date'].max()
    def avg(days): return df[df['Date'] > today - pd.Timedelta(days=days)]['PR'].mean()
    pr_7, pr_30, pr_60, pr_90, pr_365 = [avg(n) for n in [7, 30, 60, 90, 365]]
    pr_lifetime = df['PR'].mean()

    avg_box = (
        f"Average PR (last 7–d): {pr_7:.1f} %\n"
        f"Average PR (last 30–d): {pr_30:.1f} %\n"
        f"Average PR (last 60–d): {pr_60:.1f} %\n"
        f"Average PR (last 90–d): {pr_90:.1f} %\n"
        f"Average PR (last 365–d): {pr_365:.1f} %\n"
        f"\nLifetime Average PR: {pr_lifetime:.1f} %"
    )
    plt.text(df['Date'].max() - pd.Timedelta(days=300), 30, avg_box, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='black'))

    
    plt.savefig("pr_graph.png", dpi=300, bbox_inches='tight')

    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')  
    except:
        try:
            manager.full_screen_toggle()  
        except:
            pass

    plt.show()
    print("✅ Full-size graph displayed and saved as pr_graph.png")


if __name__ == "__main__":
    df = preprocess_data()
    plot_pr_graph(df)
