import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def create_saturation_heatmap(matrix, title="Matrix Heatmap"):
    """
    Creates a heatmap of an N x N matrix.
    
    Highest values have the highest saturation (dark blue).
    Lowest values have no saturation (white/light blue).
    """
    # Check if the matrix is N x N
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The input matrix must be square (N x N).")

    # Set up the plot size
    plt.figure(figsize=(10, 8))

    # --- The Heatmap Core ---
    # data: Your matrix
    # cmap: 'Blues' starts white (low) and goes to dark blue (high)
    # annot: True adds numbers to each cell (helpful for N < 20)
    # fmt: '.2f' formats those numbers to 2 decimal places
    # cbar: True shows the colorbar legend
    ax = sns.heatmap(matrix, 
                     cmap='Blues', 
                     annot=True, 
                     fmt='.2f', 
                     linewidths=0.5, 
                     cbar=True)

    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Columns (Variable X)", fontsize=12)
    plt.ylabel("Rows (Variable Y)", fontsize=12)

    # Show the plot
    plt.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    Example Usage
    --------------
    fig, ax = plt.subplots()

    im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                       cmap="YlGn", cbarlabel="harvest [t/year]")

    fig.tight_layout()
    plt.show()

    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and labels with their respective list entries
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def cov_to_corr(covariance_matrix):
    """
    Converts a covariance matrix into a correlation matrix.
    """
    # 1. Extract the diagonal (the variances)
    v = np.diag(covariance_matrix)
    
    # 2. Calculate standard deviations (sqrt of variance)
    # Adding a tiny epsilon prevents division by zero for constant variables
    std_devs = np.sqrt(v)
    
    # 3. Create the inverse diagonal matrix D^-1
    # We use 1/std_devs to get the inverse
    d_inv = np.diag(1 / std_devs)
    
    # 4. Calculate R = D^-1 * Sigma * D^-1
    correlation_matrix = d_inv @ covariance_matrix @ d_inv
    
    return correlation_matrix

def graph_oil_vs_riyal():
    # 1. Define Tickers
    # CL=F is the symbol for Crude Oil WTI Futures
    # SAR=X is the symbol for the USD/SAR exchange rate
    # tickers = ["CL=F", "NOK=X", "NGN=X", "BRL=X", "RUB=X", "CAD=X", "IQD=X"]
    #tickers = ["CL=F", "NOK=X", "NGN=X", "BRL=X", "RUB=X"]
    #tickers = ["CL=F", "NOK=X"]
    #tickers = ["CL=F", "NGN=X"]
    #tickers = ["CL=F", "BRL=X"]
    #tickers = ["CL=F", "RUB=X"]
    currency_list = [
        "SAR=X",     # Saudi Riyal (Saudi Arabia)
        "IQD=X",     # Iraqi Dinar (Iraq)
        "AED=X",     # UAE Dirham (United Arab Emirates)
        "QAR=X",     # Qatari Rial (Qatar)
        "TMT=X",     # Turkmenistani Manat (Turkmenistan)
        "CLP=X",     # Chilean Peso (Chile)
        "ZMW=X",     # Zambian Kwacha (Zambia)
        "AUDUSD=X",  # Australian Dollar (Australia)
        "MRU=X",     # Mauritanian Ouguiya (Mauritania)
        "MAD=X",     # Moroccan Dirham (Morocco)
        "GHS=X",     # Ghanaian Cedi (Ghana)
        "UZS=X",     # Uzbekistani Som (Uzbekistan)
        "BWP=X",     # Botswana Pula (Botswana)
        "XOF=X",     # CFA Franc (Côte d'Ivoire/WAEMU)
        "ETB=X",     # Ethiopian Birr (Ethiopia)
        "BRL=X",     # Brazilian Real (Brazil)
        "PYG=X"      # Paraguayan Guarani (Paraguay)
    ]

    commodity_list = [
        "CL=F",      # Crude Oil WTI Futures (Light Sweet Crude)
        "BZ=F",      # Brent Crude Oil Futures
        "NG=F",      # Natural Gas Futures
        "HG=F",      # Copper Futures
        "TIO=F",     # Iron Ore Futures (62% Fe CFR)
        "GC=F",      # Gold Futures
        "CC=F",      # Cocoa Futures
        "KC=F",      # Coffee C Futures
        "ZS=F"       # Soybean Futures
    ]

    tickers = currency_list + commodity_list
    
    # 2. Download historical data (Last 2 years)
    raw_data = yf.download(tickers, period="1y", interval="1d")['Close']
    raw_data = raw_data.dropna()
    
    # Clean data (drop missing values if markets were closed on different days)
    data = raw_data.pct_change().dropna()

    covar = np.cov(raw_data, rowvar=False)

    #create_saturation_heatmap(covar)
    corr = cov_to_corr(covar)
    create_saturation_heatmap(corr)
    fig, ax = plt.subplots()

    im, cbar = heatmap(corr, tickers, tickers, ax=ax,
                       cmap="RdBu", cbarlabel="correlation coefficient")
    # Really messy on a big matrix
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    fig.tight_layout()
    plt.show()
    
    # 3. Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Crude Oil on the left Y-axis
    color_oil = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('WTI Crude Oil Price (USD)', color=color_oil, fontsize=12)
    ax1.plot(data.index, data['CL=F'], color=color_oil, label='Crude Oil (CL=F)')
    ax1.tick_params(axis='y', labelcolor=color_oil)
    ax1.grid(True, alpha=0.3)

    for tick in tickers:
        ax1.plot(data.index, data[tick], label=tick)

    # Set Title and Layout
    plt.title('Correlation: WTI Crude Oil vs. Saudi Riyal (USD/SAR)', fontsize=16)
    fig.tight_layout()
    
    # Show the plot
    # plt.show()

if __name__ == "__main__":
    # suppress=True: Disables scientific notation
    # precision=4: Limits the output to 4 decimal places (optional)
    np.set_printoptions(suppress=True, precision=6)

    graph_oil_vs_riyal()
