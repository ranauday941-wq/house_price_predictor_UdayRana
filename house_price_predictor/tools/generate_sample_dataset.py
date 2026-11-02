"""Generate a house-price dataset that follows the schema of Kaggle's
'House Prices - Advanced Regression Techniques' (Ames, Iowa) data.

The real Kaggle file (train.csv) cannot be bundled with this project, so this
script creates a stand-in with the same kind of columns (numerical and
categorical), realistic missing values, outliers and a few duplicate records.
The web app accepts this file or the real Kaggle train.csv (target: SalePrice).

Run from the project root:   python tools/generate_sample_dataset.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1460

NEIGH = {  # name: (price effect, typical build year)
    "NridgHt": (0.32, 2000), "StoneBr": (0.30, 2002), "NoRidge": (0.34, 1996), "Somerst": (0.16, 2000),
    "CollgCr": (0.08, 1998), "Crawfor": (0.05, 1950), "Gilbert": (0.05, 2000), "NWAmes": (0.02, 1978),
    "Sawyer": (-0.08, 1962), "NAmes": (-0.06, 1955), "OldTown": (-0.20, 1925), "Edwards": (-0.16, 1950),
    "BrkSide": (-0.20, 1930), "IDOTRR": (-0.30, 1925), "MeadowV": (-0.34, 1975),
}
names = list(NEIGH)
w = np.array([4, 1.5, 1.5, 3, 10, 3, 5, 5, 5, 15, 8, 7, 4, 3, 1.2])
nb = rng.choice(names, N, p=w / w.sum())
nb_eff = np.array([NEIGH[x][0] for x in nb])
nb_year = np.array([NEIGH[x][1] for x in nb])

qual = np.clip(np.round(6 + nb_eff * 3 + rng.normal(0, 1.2, N)), 1, 10).astype(int)
cond = np.clip(np.round(rng.normal(5.6, 1.0, N)), 1, 9).astype(int)
year_built = np.clip(np.round(nb_year + rng.normal(0, 14, N)), 1875, 2010).astype(int)
remod = np.clip(year_built + np.abs(rng.normal(8, 15, N)).round(), year_built, 2010).astype(int)

gr_liv = np.exp(rng.normal(7.28 + 0.06 * (qual - 6), 0.32, N)).round().astype(int)
bsmt = np.clip((gr_liv * rng.normal(0.85, 0.22, N)).round(), 0, None).astype(int)
bsmt[rng.random(N) < 0.025] = 0
first = np.clip(bsmt + rng.normal(0, 60, N), 334, None).round().astype(int)
first[bsmt == 0] = np.clip(gr_liv[bsmt == 0] * 0.8, 334, None).round().astype(int)
lot_area = np.exp(rng.normal(9.15, 0.42, N)).round().astype(int)
lot_front = np.clip((np.sqrt(lot_area) * rng.normal(0.62, 0.12, N)).round(), 21, 313)

bedrooms = np.clip(np.round(gr_liv / 600 + rng.normal(0, 0.6, N)), 0, 8).astype(int)
full_bath = np.clip(np.round(gr_liv / 900 + rng.normal(0, 0.5, N)), 0, 3).astype(int)
half_bath = rng.choice([0, 1, 2], N, p=[0.63, 0.35, 0.02])
rooms = np.clip(bedrooms + np.round(gr_liv / 700) + 2, 2, 14).astype(int)
fire = rng.choice([0, 1, 2, 3], N, p=[0.47, 0.42, 0.10, 0.01])
cars = np.clip(np.round(gr_liv / 800 + 0.6 + rng.normal(0, 0.6, N)), 0, 4).astype(int)
garage_area = np.clip(cars * 250 + rng.normal(0, 55, N), 0, None).round().astype(int)
garage_area[cars == 0] = 0
garage_yr = np.clip(year_built + rng.integers(0, 8, N), 1900, 2010).astype(float)


def na_where(mask, values):
    """NaN where mask is True, else the (string) values."""
    return np.where(mask, np.nan, np.asarray(values, dtype=object))


def pick(options, p, size=N):
    return rng.choice(options, size, p=np.array(p) / np.sum(p))


ext_q = pick(["Ex", "Gd", "TA", "Fa"], [3, 33, 62, 2])
kit_q = pick(["Ex", "Gd", "TA", "Fa"], [7, 40, 50, 3])
ext_q = np.where(qual >= 9, "Ex", np.where(qual >= 7, "Gd", np.where(qual <= 3, "Fa", ext_q)))
kit_q = np.where(qual >= 9, "Ex", np.where(qual >= 7, "Gd", np.where(qual <= 3, "Fa", kit_q)))
heat_q = pick(["Ex", "Gd", "TA", "Fa", "Po"], [50, 16, 29, 5, 0.1])
central_air = pick(["Y", "N"], [93, 7])
zoning = pick(["RL", "RM", "FV", "RH", "C (all)"], [79, 15, 4.5, 1, 0.7])
house_style = pick(["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "1.5Unf"], [50, 30, 10, 4.5, 2.5, 3])
bldg_type = pick(["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"], [83, 8, 3.5, 3, 2.5])
paved = pick(["Y", "N", "P"], [92, 3, 5])
sale_cond = pick(["Normal", "Abnorml", "Partial", "Family", "Alloca"], [82, 7, 9, 1.4, 0.8])
mo_sold = rng.integers(1, 13, N)
yr_sold = rng.choice([2006, 2007, 2008, 2009, 2010], N)
mas_vnr = np.where(rng.random(N) < 0.55, 0, np.exp(rng.normal(4.9, 0.9, N))).round()

premium = {"Ex": 0.11, "Gd": 0.04, "TA": 0.0, "Fa": -0.07, "Po": -0.1}
log_price = (
    11.93
    + 0.085 * (qual - 6) + 0.020 * (cond - 5)
    + 0.00031 * (gr_liv - 1500) + 0.00014 * (bsmt - 1000)
    + 0.048 * cars + 0.0035 * (year_built - 1971) + 0.0018 * (remod - 1985)
    + 0.000006 * np.clip(lot_area - 10000, -6000, 25000)
    + 0.028 * fire + 0.020 * full_bath + 0.35 * nb_eff
    + np.vectorize(premium.get)(kit_q) + np.vectorize(premium.get)(ext_q) * 0.7
    + np.where(central_air == "Y", 0.05, -0.03)
    + np.where(sale_cond == "Partial", 0.05, np.where(sale_cond == "Abnorml", -0.07, 0))
    + rng.normal(0, 0.105, N)
)
price = np.clip(np.exp(log_price), 36000, None).round(-2).astype(int)

df = pd.DataFrame({
    "Id": np.arange(1, N + 1), "MSZoning": zoning, "Neighborhood": nb, "BldgType": bldg_type,
    "HouseStyle": house_style, "LotFrontage": lot_front, "LotArea": lot_area, "OverallQual": qual,
    "OverallCond": cond, "YearBuilt": year_built, "YearRemodAdd": remod, "MasVnrArea": mas_vnr,
    "ExterQual": ext_q, "TotalBsmtSF": bsmt, "1stFlrSF": first, "GrLivArea": gr_liv,
    "FullBath": full_bath, "HalfBath": half_bath, "BedroomAbvGr": bedrooms, "KitchenQual": kit_q,
    "TotRmsAbvGrd": rooms, "Fireplaces": fire, "HeatingQC": heat_q, "CentralAir": central_air,
    "GarageYrBlt": garage_yr, "GarageCars": cars, "GarageArea": garage_area, "PavedDrive": paved,
    "MoSold": mo_sold, "YrSold": yr_sold, "SaleCondition": sale_cond, "SalePrice": price,
})
df["GarageType"] = na_where(cars == 0, pick(["Attchd", "Detchd", "BuiltIn", "Basment"], [60, 27, 6, 7]))
df["FireplaceQu"] = na_where(fire == 0, pick(["Gd", "TA", "Fa", "Ex", "Po"], [48, 40, 5, 3, 3]))
df["Alley"] = na_where(rng.random(N) >= 0.062, pick(["Grvl", "Pave"], [50, 50]))
df["PoolQC"] = na_where(rng.random(N) >= 0.005, ["Gd"] * N)
df["Fence"] = na_where(rng.random(N) >= 0.19, pick(["MnPrv", "GdWo", "GdPrv", "MnWw"], [55, 25, 15, 5]))
df["MiscFeature"] = na_where(rng.random(N) >= 0.04, pick(["Shed", "Gar2", "Othr"], [90, 5, 5]))
df["Electrical"] = "SBrkr"

# --- make the data "dirty" like a real-world file -----------------------------
df.loc[rng.random(N) < 0.177, "LotFrontage"] = np.nan          # ~17.7 % missing
df.loc[cars == 0, "GarageYrBlt"] = np.nan                      # houses without a garage
df.loc[rng.random(N) < 0.006, "MasVnrArea"] = np.nan
df.loc[int(rng.integers(0, N)), "Electrical"] = np.nan
# genuine extreme records (very large houses, huge lots)
idx = rng.choice(N, 5, replace=False)
df.loc[idx[:3], "GrLivArea"] = [4676, 5642, 4316]
df.loc[idx[:3], "SalePrice"] = [184750, 160000, 755000]
df.loc[idx[3:], "LotArea"] = [215245, 159000]
# duplicates (same house entered twice with a new Id)
dups = df.sample(20, random_state=1).copy()
dups["Id"] = np.arange(N + 1, N + 21)
df = pd.concat([df, dups], ignore_index=True)
df = df.sample(frac=1, random_state=3).reset_index(drop=True)

df.to_csv("sample_data/house_prices_sample.csv", index=False)
print("saved", df.shape, "median price", int(df["SalePrice"].median()))
