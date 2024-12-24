# %% [markdown]
# # Plot some categories from the Steam Hardware Survey dataset



# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype

%config InlineBackend.figure_format = 'retina'

# %%


# %%
df = pd.read_parquet("steam_hw_survey copy 2.parquet")
df.dtypes
df["category"] = df["category"].astype("category")
df["platform"] = df["platform"].astype("category")
df

# %% [markdown]
# List categories throughout the years, some may be specific to a platform, and others are only available for some months or years.

# %%
df["category"].cat.categories

# %%
df.shape

# %% [markdown]
# ## Share of Operating Systems

# %%
df_os_version = df.loc[(df["category"] == "OS Version (total)") & (df["perc"] > 1)].copy()
df_os_version["year"] = df_os_version["date"].dt.year.astype(str)
df_os_version

# %%
df_os_version = (
    df_os_version
    .groupby(["year", "index"])
    .mean()
    .reset_index()
)
df_os_version.head(5)

# %%
_, ax = plt.subplots(nrows=1, figsize=(10, 5), constrained_layout=True)
ax.set(xlabel="Date", ylabel="Percentage", ylim=[0, 100])
sns.histplot(
    df_os_version,
    x="year",
    hue="index",
    weights="perc",
    multiple="dodge",
    discrete=True,
    hue_order=["Windows", "OSX", "Linux"], 
    palette="muted",
    shrink=0.75,
    linewidth=0.0,
    ax=ax,
)
sns.move_legend(
    ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False
)

# %%
_, ax = plt.subplots(nrows=1, figsize=(10, 5), constrained_layout=True)
ax.set(xlabel="Date", ylabel="Percentage", ylim=[0, 100])
sns.histplot(
    df_os_version,
    x="year",
    hue="index",
    weights="perc",
    multiple="stack",
    discrete=True,
    hue_order=["Windows", "OSX", "Linux"],
    palette="muted",
    shrink=0.75,
    linewidth=0.0,
    ax=ax,
)
sns.move_legend(
    ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False
)

# %%
df_windows_version = df.loc[(df["category"] == "OS Version") & (df["perc"] > 1) & df["index"].str.contains("Windows", case=False)].copy()
df_windows_version["year"] = df_windows_version["date"].dt.year.astype(str)
df_windows_version

# %%

df_windows_version['index'] = df_windows_version['index'].replace({
    'Windows XP 32 bit': 'Windows XP',
    'Windows XP 64 bit': 'Windows XP',
    'Windows 7 64 bit': 'Windows 7',
    'Windows Vista 32 bit': 'Windows Vista',
    'Windows Vista 64 bit': 'Windows Vista',
    'Windows 8 64 bit': 'Windows 8',
    'Windows 8.1 64 bit': 'Windows 8.1',
    'Windows 10 64 bit': 'Windows 10',
    'Windows 11 64 bit': 'Windows 11'
})
# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_windows_version,
    x="year",
    y="perc",
    hue="index",  # Assuming "index" represents CPU speed categories
    palette="tab10"
)
plt.title("Distribution of Windows Versions", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="CPU Speed Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Share of CPU manufacturers

# %%
df_cpu_vendor = df.loc[(df["category"] == "Processor Vendor") & (df["perc"] > 1)].copy()
df_cpu_vendor["year"] = df_cpu_vendor["date"].dt.year.astype(str)
df_cpu_vendor

# %%
df_cpu_vendor_pc = (
    df_cpu_vendor.loc[df_cpu_vendor["platform"] == "pc"]
    .groupby(["year", "index"])
    .mean()
    .reset_index()
)
df_cpu_vendor_pc.head(5)

# %%
_, ax = plt.subplots(nrows=1, figsize=(10, 5), constrained_layout=True)
ax.set(xlabel="Date", ylabel="Percentage", ylim=[0, 100])
sns.histplot(
    df_cpu_vendor_pc,
    x="year",
    hue="index",
    weights="perc",
    multiple="dodge",
    discrete=True,
    hue_order=["GenuineIntel", "AuthenticAMD"],
    palette="muted",
    shrink=0.75,
    linewidth=0.0,
    ax=ax,
)
sns.move_legend(
    ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False
)

# %% [markdown]
# Same plot with stacked barchart:

# %%
_, ax = plt.subplots(nrows=1, figsize=(10, 5), constrained_layout=True)
ax.set(xlabel="Date", ylabel="Percentage", ylim=[0, 100])
sns.histplot(
    df_cpu_vendor_pc,
    x="year",
    hue="index",
    weights="perc",
    multiple="stack",
    discrete=True,
    hue_order=["GenuineIntel", "AuthenticAMD"],
    palette="muted",
    shrink=0.75,
    linewidth=0.0,
    ax=ax,
)
sns.move_legend(
    ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False
)

# %% [markdown]
# Monthly share of CPUs with platform information.

# %%
_, ax = plt.subplots(nrows=1, figsize=(10, 5), constrained_layout=True)
sns.lineplot(
    x="date",
    y="perc",
    hue="index",
    style="platform",
    data=df_cpu_vendor,
    style_order=["pc", "mac", "linux"],
)
ax.set(title="CPU Share by Vendor", xlabel="Date", ylabel="Percentage", ylim=[0, 100])
sns.move_legend(ax, "lower left", ncol=2)

# %%
df_cpu_vendor = (
    df_cpu_vendor.groupby(["index", df_cpu_vendor["date"].dt.year, "platform"])
    .mean()
    .reset_index()
)
df_cpu_vendor.pivot(index=["platform", "index"], columns="date", values="perc")

# %% [markdown]
# ## CPU core count

# %%
df_num_cpu = df.loc[
    (df["category"] == "Physical CPUs") & (df["index"] != "Unspecified")
].copy()
df_num_cpu["num_cpus"] = df_num_cpu["index"].str.strip("cpus").astype(int)
df_num_cpu.loc[df_num_cpu["num_cpus"] > 16, "index"] = ">16 cpus"
df_num_cpu.drop(columns=["num_cpus"], inplace=True)
df_num_cpu

# %%
cat_num_cpu = CategoricalDtype(
    [
        "1 cpu",
        "2 cpus",
        # "3 cpus",
        "4 cpus",
        "6 cpus",
        "8 cpus",
        "10 cpus",
        "12 cpus",
        "16 cpus",
        ">16 cpus",
    ],
    ordered=True,
)
df_num_cpu["index"] = df_num_cpu["index"].astype(cat_num_cpu)

# %%
# sum all '+16 cpus' percentages for each month
df_num_cpu = (
    df_num_cpu.groupby(["index", "date", "platform"], as_index=False)
    .sum()
    .replace(0, np.nan)
)
# compute mean annual share
df_num_cpu = (
    df_num_cpu.groupby(["index", df_num_cpu["date"].dt.year, "platform"])
    .mean()
    .reset_index()
)
df_num_cpu["date"] = df_num_cpu["date"].astype(str)

# %%
df_normalized = df_num_cpu.copy()
df_normalized["perc"] = (
    df_normalized.groupby(["platform", "date"])["perc"]
    .transform(lambda x: (x / x.sum()) * 100)
)

# Plotting
_, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(20, 10),
    constrained_layout=True,
    sharey=True,
    sharex=True,
)

for ax, subset in zip(axes.flat, ["pc", "combined", "mac", "linux"]):
    sns.histplot(
        df_normalized.loc[df_normalized["platform"] == subset],
        x="date",
        hue="index",
        weights="perc",
        multiple="stack",
        discrete=True,
        palette="muted",
        shrink=0.75,
        linewidth=0.0,
        ax=ax,
        legend=True if subset == "combined" else False,
    )
    ax.set(title=f"{subset}", xlabel="Date", ylabel="Percentage", ylim=[0, 100])
    if subset == "combined":
        sns.move_legend(
            ax,
            "upper left",
            bbox_to_anchor=(0, 1),
            title=None,
            frameon=False,
        )

plt.show()

# %%
df_num_cpu.pivot(index=["platform", "index"], columns="date", values="perc")

# %% [markdown]
# ## Linux flavors

# %%
df_linux = df.loc[(df["category"] == "Linux Version")].copy()
df_linux

# %%
df_linux["index"] = (
    df_linux["index"]
    .replace(to_replace=r" \d.*$", value="", regex=True)
    .astype("category")
)
# sum the cleaned distro shares
df_linux = df_linux.groupby(["index", "date"], as_index=False).sum()
# compute mean annual share
df_linux = df_linux.groupby(["index", df_linux["date"].dt.year]).mean().reset_index()
df_linux["date"] = df_linux["date"].astype(str)

# %%
# Combine different Freedesktop.org names into a single category
df_linux_combined = df_linux.copy()
df_linux_combined["index"] = df_linux_combined["index"].replace({
    "Freedesktop SDK": "Freedesktop.org",
    "Freedesktop.org SDK": "Freedesktop.org",
    "Description:Freedesktop.org": "Freedesktop.org",
    '"Arch Linux"' : "Arch Linux",
    '"Manjaro Linux"' : "Manjaro Linux",
    '"SteamOS Holo"' : "SteamOS"
})

# Plotting the combined data
_, ax = plt.subplots(nrows=1, figsize=(10, 7), constrained_layout=True)
sns.histplot(
    df_linux_combined,
    x="date",
    hue="index",
    weights="perc",
    multiple="stack",
    palette="tab20",
    discrete=True,
    shrink=0.75,
    linewidth=0.0,
    ax=ax,
)
ax.set(title="Linux Distributions", xlabel="Date", ylabel="Percentage", ylim=[0, 100])
sns.move_legend(ax, "lower left", bbox_to_anchor=(1, 0.5), title=None)

plt.show()


# %%
df_linux.pivot(index=["index"], columns="date", values="perc").replace(0, np.nan)

# %% [markdown]
# The category `Other` shadows many distribution versions, so be careful making assumptions with these values. Another source of distribution usage is [All Roads Lead to Arch: The Evolution of Linux Distros Used for Gaming Over Time](https://boilingsteam.com/all-roads-lead-to-arch-the-evolution-of-linux-distros-used-for-gaming-over-time), although the series begins at the end of 2018.

# %% [markdown]
# ## Share of GPU manufacturers

# %%
# there are some errors for data collected in 2015 showing values of 'Other' > 90%
# in 2014, AMD enters the series and average values are higher than they should be
df_gpu_vendor = df.loc[
    (df["category"] == "Video Card Description") & (df["perc"] < 90)
].copy()
df_gpu_vendor["year"] = df_gpu_vendor["date"].dt.year.astype(str)
df_gpu_vendor

# %%
group_vendor = lambda x: df_gpu_vendor["index"].str.contains(x, case=False)

df_gpu_vendor.loc[group_vendor("AMD|Radeon"), "gpu_vendor"] = "AMD"
df_gpu_vendor.loc[group_vendor("ATI|ATI Radeon"), "gpu_vendor"] = "ATI"
df_gpu_vendor.loc[group_vendor("Intel"), "gpu_vendor"] = "Intel"
df_gpu_vendor.loc[group_vendor("NVIDIA|GeForce"), "gpu_vendor"] = "NVIDIA"
df_gpu_vendor.loc[group_vendor("Apple"), "gpu_vendor"] = "Apple"
df_gpu_vendor.loc[group_vendor("Other"), "gpu_vendor"] = "Other"
df_gpu_vendor["gpu_vendor"] = df_gpu_vendor["gpu_vendor"].replace(np.nan, "Other")

# %%
cat_gpu_vendor = CategoricalDtype(
    [
        "AMD",
        "Apple",
        "ATI",
        "Intel",
        "NVIDIA",
        "Other",
    ],
    ordered=True,
)
df_gpu_vendor["gpu_vendor"] = df_gpu_vendor["gpu_vendor"].astype(cat_gpu_vendor)

# %%
df_gpu_vendor = (
    df_gpu_vendor.groupby(["gpu_vendor", "date", "platform"], as_index=False)
    .sum()
    .replace(0, np.nan)
)
# compute mean annual share
df_gpu_vendor = (
    df_gpu_vendor.groupby(["gpu_vendor", df_gpu_vendor["date"].dt.year, "platform"])
    .mean()
    .reset_index()
)
df_gpu_vendor["date"] = df_gpu_vendor["date"].astype(str)

# %%
df_normalized_gpu_vendor = df_gpu_vendor.copy()
df_normalized_gpu_vendor["perc"] = (
    df_normalized_gpu_vendor.groupby(["platform", "date"])["perc"]
    .transform(lambda x: (x / x.sum()) * 100)
)

_, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(20, 10),
    constrained_layout=True,
    sharey=True,
    sharex=True,
)
for ax, subset in zip(axes.flat, ["pc", "combined", "mac", "linux"]):
    sns.histplot(
        df_normalized_gpu_vendor.loc[df_normalized_gpu_vendor["platform"] == subset],
        x="date",
        hue="gpu_vendor",
        weights="perc",
        multiple="stack",
        discrete=True,
        palette="muted",
        shrink=0.75,
        linewidth=0.0,
        ax=ax,
        legend=True if subset == "combined" else False,
    )
    ax.set(title=f"{subset}", xlabel="Date", ylabel="Percentage", ylim=[0, 100])
    if subset == "combined":
        sns.move_legend(
            ax,
            "upper left",
            bbox_to_anchor=(0, 1),
            title=None,
            frameon=False,
        )

# %%
df_gpu_vendor.pivot(index=["platform", "gpu_vendor"], columns="date", values="perc")

# %%
df_system_ram = df.loc[
    (df["category"] == "Intel CPU Speeds") & 
    (df["platform"] == "combined")&
    (df["perc"] > 1) & 
    (df["date"].dt.year)  # Filter for years between 2010 and 2020
].copy()

df_system_ram["year"] = df_system_ram["date"].dt.year.astype(str)

df_system_ram


# %%
df_system_ram = df.loc[
    (df["category"] == "Intel CPU Speeds") &
    (df["platform"] == "combined") &
    (df["perc"] > 1)
].copy()

df_system_ram["year"] = df_system_ram["date"].dt.year  # Convert to integer for filtering

# Filter for alternate years
alternate_years = df_system_ram["year"].unique()[::2]  # Select every second year
df_system_ram = df_system_ram[df_system_ram["year"].isin(alternate_years)]
df_system_ram["year"] = df_system_ram["year"].astype(str)  # Convert back to string for plotting

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_system_ram,
    x="year",
    y="perc",
    hue="index",  # Assuming "index" represents CPU speed categories
    palette="tab10"
)
plt.title("Distribution of CPU Speeds (Alternate Years)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="CPU Speed Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


