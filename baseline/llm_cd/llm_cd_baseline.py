from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from baseline.llm_cd.baseline_datasets import (
    load_adult_baseline_dataset,
    load_cardio_baseline_dataset,
    load_creditcard_baseline_dataset,
    load_crime_baseline_dataset,
    load_diamonds_baseline_dataset,
    load_elevator_baseline_dataset,
    load_housesale_baseline_dataset,
    load_meps_baseline_dataset,
)
from baseline.llm_cd.llm_cd_prompts import (
    conditional_independence_prompt,
    cycle_break_prompt,
    directed_edge_prompt,
    undirected_edge_prompt,
)


DATASET_LOADERS = {
    "adult": load_adult_baseline_dataset,
    "cardio": load_cardio_baseline_dataset,
    "creditcard": load_creditcard_baseline_dataset,
    "crime": load_crime_baseline_dataset,
    "diamonds": load_diamonds_baseline_dataset,
    "elevator": load_elevator_baseline_dataset,
    "housesale": load_housesale_baseline_dataset,
    "meps": load_meps_baseline_dataset,
}


@dataclass
class GraphDiscoveryResult:
    parents: List[int]
    graph: np.ndarray
    feature_names: List[str]
    feature_descriptions: Dict[str, str]
    pc_directed_edges: List[Tuple[int, int]]
    pc_undirected_edges: List[Tuple[int, int]]
    llm_calls: int
    llm_cache_hits: int


@dataclass
class BaselineResult:
    dataset: str
    task_type: str
    target_idx: int
    target_name: str
    parents: List[int]
    parent_names: List[str]
    graph: np.ndarray
    metrics: Dict[str, Dict[str, float]]
    llm_calls: int
    llm_cache_hits: int


def load_baseline_arrays(dataset_name: str):
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choices: {sorted(DATASET_LOADERS)}")
    dataset_obj, v, num_classes_dict, target_idx = DATASET_LOADERS[dataset_name]()
    x = dataset_obj.x.detach().cpu().numpy().astype(float)
    y = dataset_obj.y.detach().cpu().numpy().reshape(-1)
    full = x.copy()
    full[:, target_idx] = y
    if not isinstance(v, np.ndarray):
        v = np.asarray(v)
    return full, y, v.astype(int), num_classes_dict, target_idx


def feature_metadata(dataset_name: str, n_features: int) -> Tuple[List[str], Dict[str, str]]:
    names: List[str]
    descriptions: Dict[str, str]
    if dataset_name == "adult":
        names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "income",
        ][:n_features]
        descriptions = {
            name: f"Adult Census feature: {name.replace('_', ' ')}." for name in names
        }
        if "income" in descriptions:
            descriptions["income"] = "Whether annual income is above the dataset income threshold."
    elif dataset_name == "creditcard":
        names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        names = names[:n_features]
        descriptions = {
            "Time": "Elapsed time from the first transaction in the dataset.",
            "Amount": "Transaction amount.",
            "Class": "Fraud indicator where the positive class indicates fraudulent transaction.",
        }
        for name in names:
            descriptions.setdefault(name, f"An anonymized PCA transaction feature named {name}.")
    elif dataset_name == "diamonds":
        names = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"][:n_features]
        descriptions = {
            "carat": "Diamond weight in carats.",
            "cut": "Diamond cut quality.",
            "color": "Diamond color grade.",
            "clarity": "Diamond clarity grade.",
            "depth": "Diamond depth percentage.",
            "table": "Diamond table percentage.",
            "x": "Diamond length dimension.",
            "y": "Diamond width dimension.",
            "z": "Diamond depth dimension.",
            "price": "Diamond price.",
        }
    elif dataset_name == "elevator":
        names = _read_header("raw_data/elevator.csv", n_features) or [
            "revolutions",
            "humidity",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "vibration",
        ][:n_features]
        descriptions = {
            name: f"Elevator predictive maintenance variable: {name}." for name in names
        }
        if "vibration" in descriptions:
            descriptions["vibration"] = "Elevator vibration measurement, used as the prediction target."
    elif dataset_name == "housesale":
        names = _read_header("raw_data/housesale.csv", n_features) or [f"feature_{i}" for i in range(n_features)]
        descriptions = {name: f"House sale attribute: {name}." for name in names}
        for key in ("Price", "price"):
            if key in descriptions:
                descriptions[key] = "House sale price, used as the prediction target."
    elif dataset_name == "cardio":
        names = [
            "gender",
            "height_bin",
            "weight_bin",
            "systolic_pressure_bin",
            "diastolic_pressure_bin",
            "age",
            "cholesterol",
            "glucose",
            "smoking",
            "alcohol_intake",
            "physical_activity",
            "cardio",
        ][:n_features]
        descriptions = {name: f"Cardiovascular health variable: {name.replace('_', ' ')}." for name in names}
        if "cardio" in descriptions:
            descriptions["cardio"] = "Cardiovascular disease label."
    elif dataset_name == "crime":
        names = _read_header("raw_data/crime.csv", n_features) or [f"feature_{i}" for i in range(n_features)]
        _crime_descriptions = {
            "population": "Total population of the community.",
            "householdsize": "Average number of people per household in the community.",
            "racepctblack": "Percentage of population that is African American / Black.",
            "racePctWhite": "Percentage of population that is Caucasian / White.",
            "racePctAsian": "Percentage of population that is Asian.",
            "racePctHisp": "Percentage of population that is Hispanic.",
            "agePct12t21": "Percentage of population aged 12 to 21.",
            "agePct12t29": "Percentage of population aged 12 to 29.",
            "agePct16t24": "Percentage of population aged 16 to 24.",
            "agePct65up": "Percentage of population aged 65 and over.",
            "numbUrban": "Number of people living in areas classified as urban.",
            "pctUrban": "Percentage of population living in urban areas.",
            "medIncome": "Median household income in the community.",
            "pctWWage": "Percentage of households with wage or salary income.",
            "pctWFarmSelf": "Percentage of households with farm or self-employment income.",
            "pctWInvInc": "Percentage of households with investment or rent income.",
            "pctWSocSec": "Percentage of households with Social Security income.",
            "pctWPubAsst": "Percentage of households with public assistance income.",
            "pctWRetire": "Percentage of households with retirement income.",
            "medFamInc": "Median family income.",
            "perCapInc": "Per capita income.",
            "whitePerCap": "Per capita income for White population.",
            "blackPerCap": "Per capita income for African American / Black population.",
            "indianPerCap": "Per capita income for Native American population.",
            "AsianPerCap": "Per capita income for Asian population.",
            "OtherPerCap": "Per capita income for other race categories.",
            "HispPerCap": "Per capita income for Hispanic population.",
            "NumUnderPov": "Number of people under the poverty level.",
            "PctPopUnderPov": "Percentage of population under the poverty level.",
            "PctLess9thGrade": "Percentage of people with less than 9th grade education.",
            "PctNotHSGrad": "Percentage of people without a high school diploma.",
            "PctBSorMore": "Percentage of people with a bachelor's degree or higher.",
            "PctUnemployed": "Percentage of people aged 16 and over who are unemployed.",
            "PctEmploy": "Percentage of people aged 16 and over who are employed.",
            "PctEmplManu": "Percentage of employed people in manufacturing.",
            "PctEmplProfServ": "Percentage of employed people in professional services.",
            "PctOccupManu": "Percentage of employed people in manufacturing occupations.",
            "PctOccupMgmtProf": "Percentage of employed people in management or professional occupations.",
            "MalePctDivorce": "Percentage of males who are divorced.",
            "MalePctNevMarr": "Percentage of males who have never married.",
            "FemalePctDiv": "Percentage of females who are divorced.",
            "TotalPctDiv": "Percentage of total population who are divorced.",
            "PersPerFam": "Average number of persons per family.",
            "PctFam2Par": "Percentage of families with both parents present.",
            "PctKids2Par": "Percentage of children in families with both parents present.",
            "PctYoungKids2Par": "Percentage of young children in two-parent families.",
            "PctTeen2Par": "Percentage of teenagers in two-parent families.",
            "PctWorkMomYoungKids": "Percentage of moms of young children in the labor force.",
            "PctWorkMom": "Percentage of mothers in the labor force.",
            "NumIlleg": "Number of children born to unmarried women.",
            "PctIlleg": "Percentage of children born to unmarried women.",
            "NumImmig": "Total number of immigrants living in the community.",
            "PctImmigRecent": "Percentage of immigrants who immigrated within the last 3 years.",
            "PctImmigRec5": "Percentage of immigrants who immigrated within the last 5 years.",
            "PctImmigRec8": "Percentage of immigrants who immigrated within the last 8 years.",
            "PctImmigRec10": "Percentage of immigrants who immigrated within the last 10 years.",
            "PctRecentImmig": "Percentage of recent immigrants (within last 3 years).",
            "PctRecImmig5": "Percentage of recent immigrants (within last 5 years).",
            "PctRecImmig8": "Percentage of recent immigrants (within last 8 years).",
            "PctRecImmig10": "Percentage of recent immigrants (within last 10 years).",
            "PctSpeakEnglOnly": "Percentage of people who speak only English.",
            "PctNotSpeakEnglWell": "Percentage of people who do not speak English well.",
            "PctLargHouseFam": "Percentage of families living in large households.",
            "PctLargHouseOccup": "Percentage of occupied housing units that are large households.",
            "PersPerOccupHous": "Average number of persons per occupied housing unit.",
            "PersPerOwnOccHous": "Average number of persons per owner-occupied housing unit.",
            "PersPerRentOccHous": "Average number of persons per rental-occupied housing unit.",
            "PctPersOwnOccup": "Percentage of people in owner-occupied housing units.",
            "PctPersDenseHous": "Percentage of people in dense housing (>1 person per room).",
            "PctHousLess3BR": "Percentage of housing units with fewer than 3 bedrooms.",
            "MedNumBR": "Median number of bedrooms per housing unit.",
            "HousVacant": "Number of vacant housing units.",
            "PctHousOccup": "Percentage of housing units that are occupied.",
            "PctHousOwnOcc": "Percentage of occupied housing units that are owner-occupied.",
            "PctVacantBoarded": "Percentage of vacant housing units that are boarded up.",
            "PctVacMore6Mos": "Percentage of vacant housing units vacant more than 6 months.",
            "MedYrHousBuilt": "Median year housing units were built.",
            "PctHousNoPhone": "Percentage of occupied housing units without phone service.",
            "PctWOFullPlumb": "Percentage of housing units without complete plumbing facilities.",
            "OwnOccLowQuart": "Lower quartile value of owner-occupied housing units.",
            "OwnOccMedVal": "Median value of owner-occupied housing units.",
            "OwnOccHiQuart": "Upper quartile value of owner-occupied housing units.",
            "RentLowQ": "Lower quartile of contract rent.",
            "RentMedian": "Median contract rent.",
            "RentHighQ": "Upper quartile of contract rent.",
            "MedRent": "Median gross rent.",
            "MedRentPctHousInc": "Median gross rent as a percentage of household income.",
            "MedOwnCostPctInc": "Median owner cost as a percentage of household income (with mortgage).",
            "MedOwnCostPctIncNoMtg": "Median owner cost as a percentage of household income (without mortgage).",
            "NumInShelters": "Number of people in homeless shelters.",
            "NumStreet": "Number of homeless people counted in the street.",
            "PctForeignBorn": "Percentage of people who are foreign-born.",
            "PctBornSameState": "Percentage of people born in the same state as current residence.",
            "PctSameHouse85": "Percentage of people living in the same house as in 1985.",
            "PctSameCity85": "Percentage of people living in the same city as in 1985.",
            "PctSameState85": "Percentage of people living in the same state as in 1985.",
            "LemasSwornFT": "Number of full-time sworn police officers (LEMAS survey).",
            "LemasSwFTPerPop": "Full-time sworn police officers per 100k population.",
            "LemasSwFTFieldOps": "Number of full-time sworn officers assigned to field operations.",
            "LemasSwFTFieldPerPop": "Field operations officers per 100k population.",
            "LemasTotalReq": "Total requests for police service.",
            "LemasTotReqPerPop": "Total requests for police service per population.",
            "PolicReqPerOffic": "Police requests per officer.",
            "PolicPerPop": "Number of police officers per 100k population.",
            "RacialMatchCommPol": "Racial match between community and police force (index).",
            "PctPolicWhite": "Percentage of police officers who are White.",
            "PctPolicBlack": "Percentage of police officers who are African American / Black.",
            "PctPolicHisp": "Percentage of police officers who are Hispanic.",
            "PctPolicAsian": "Percentage of police officers who are Asian.",
            "PctPolicMinor": "Percentage of police officers who are racial/ethnic minorities.",
            "OfficAssgnDrugUnits": "Number of officers assigned to special drug units.",
            "NumKindsDrugsSeiz": "Number of different kinds of drugs seized by police.",
            "PolicAveOTWorked": "Average overtime hours worked by police officers.",
            "LandArea": "Land area of the community in square miles.",
            "PopDens": "Population density per square mile.",
            "PctUsePubTrans": "Percentage of people using public transit for commuting.",
            "PolicCars": "Number of police cars in the community.",
            "PolicOperBudg": "Police operating budget.",
            "LemasPctPolicOnPatr": "Percentage of police officers on patrol duty.",
            "LemasGangUnitDeploy": "LEMAS indicator: gang unit deployed (0 or 1).",
            "LemasPctOfficDrugUn": "Percentage of police officers assigned to drug units.",
            "PolicBudgPerPop": "Police operating budget per population.",
        }
        descriptions = {}
        for name in names:
            descriptions[name] = _crime_descriptions.get(name, f"UCI Communities and Crime variable: {name}.")
        target_name_crime = names[-1] if names else "ViolentCrimesPerPop"
        descriptions[target_name_crime] = "Violent crimes per population (incidents per 100k), used as the regression prediction target."
    elif dataset_name == "meps":
        names = _read_meps_header(n_features)
        _meps_desc_map = {
            "AGE": "Age of the survey respondent.",
            "RACE": "Race/ethnicity of the respondent (categorical, one-hot encoded).",
            "PCS42": "Physical Component Summary score (SF-12), measuring physical health status. Continuous.",
            "MCS42": "Mental Component Summary score (SF-12), measuring mental health status. Continuous.",
            "K6SUM42": "Kessler 6 (K6) psychological distress scale total score, range 0-24. Continuous.",
            "UTILIZATION_reg": "Healthcare utilization measure (total number of medical visits/expenditures), used as the regression prediction target.",
        }
        descriptions = {}
        for name in names:
            if name in _meps_desc_map:
                descriptions[name] = _meps_desc_map[name]
            elif "=" in name:
                var_name = name.split("=")[0].strip()
                _meps_var_meanings = {
                    "REGION": "Census region of residence (1=Northeast, 2=Midwest, 3=South, 4=West).",
                    "SEX": "Sex of the respondent (1=Male, 2=Female).",
                    "MARRY": "Marital status of the respondent (1=Married, others=single/widowed/divorced/separated).",
                    "FTSTU": "Full-time student status (-1=Inapplicable, 1=Full-time, 2=Part-time, 3=Not a student).",
                    "ACTDTY": "Active duty military status (1-4, different duty categories).",
                    "HONRDC": "Honorably discharged from military (1-4 categories).",
                    "RTHLTH": "Self-rated health status (-1=Inapplicable, 1=Excellent, ..., 5=Poor).",
                    "MNHLTH": "Self-rated mental health status (-1=Inapplicable, 1=Excellent, ..., 5=Poor).",
                    "HIBPDX": "High blood pressure diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "CHDDX": "Coronary heart disease diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "ANGIDX": "Angina diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "MIDX": "Heart attack / myocardial infarction diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "OHRTDX": "Other heart disease diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "STRKDX": "Stroke diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "EMPHDX": "Emphysema diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "CHBRON": "Chronic bronchitis diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "CHOLDX": "High cholesterol diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "CANCERDX": "Cancer diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "DIABDX": "Diabetes diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "JTPAIN": "Joint pain in last 12 months (-1=Inapplicable, 1=Yes, 2=No).",
                    "ARTHDX": "Arthritis diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "ARTHTYPE": "Type of arthritis (-1=Inapplicable, 1=Rheumatoid, 2=Osteoarthritis, 3=Other).",
                    "ASTHDX": "Asthma diagnosis (1=Yes, 2=No).",
                    "ADHDADDX": "Attention deficit hyperactivity disorder / ADD diagnosis (-1=Inapplicable, 1=Yes, 2=No).",
                    "PREGNT": "Pregnancy status (-1=Inapplicable, 1=Yes, 2=No).",
                    "WLKLIM": "Walking limitation (-1=Inapplicable, 1=Yes, 2=No).",
                    "ACTLIM": "Activity limitation (-1=Inapplicable, 1=Yes, 2=No).",
                    "SOCLIM": "Social limitation (-1=Inapplicable, 1=Yes, 2=No).",
                    "COGLIM": "Cognitive limitation (-1=Inapplicable, 1=Yes, 2=No).",
                    "DFHEAR42": "Hearing difficulty (-1=Inapplicable, 1=Yes, 2=No).",
                    "DFSEE42": "Vision difficulty (-1=Inapplicable, 1=Yes, 2=No).",
                    "ADSMOK42": "Currently smoke cigarettes (-1=Inapplicable, 1=Yes, 2=No).",
                    "PHQ242": "PHQ-2 depression screener score (0-6, higher = more depressed mood).",
                    "EMPST": "Employment status (-1=Inapplicable, 1=Employed, 2=Not employed, 3=Not in labor force, 4=Unknown).",
                    "POVCAT": "Poverty category (1=Poor/Negative, 2=Near poor, 3=Low income, 4=Middle income, 5=High income).",
                    "INSCOV": "Health insurance coverage (1=Private, 2=Public only, 3=Uninsured).",
                }
                descriptions[name] = _meps_var_meanings.get(var_name, f"MEPS survey one-hot encoded variable: {var_name}.")
            else:
                descriptions[name] = f"MEPS survey variable: {name}."
        target_name_meps = names[-1] if names else "UTILIZATION_reg"
        descriptions[target_name_meps] = "Healthcare utilization measure (total number of medical visits/expenditures), used as the regression prediction target."
    else:
        names = [f"feature_{i}" for i in range(n_features)]
        descriptions = {name: f"Tabular feature {i}." for i, name in enumerate(names)}

    if len(names) < n_features:
        names = names + [f"feature_{i}" for i in range(len(names), n_features)]
    names = _dedupe_names(names[:n_features])
    descriptions = {name: descriptions.get(name, name) for name in names}
    return names, descriptions


def _read_header(path: str, n_features: int) -> Optional[List[str]]:
    file_path = Path(path)
    if not file_path.exists():
        return None
    cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    if len(cols) != n_features:
        return None
    return [str(col) for col in cols]


def _read_meps_header(n_features: int) -> List[str]:
    """Read MEPS CSV header and filter out dropped columns (Unnamed, PERWT)."""
    file_path = Path("raw_data/meps.csv")
    if not file_path.exists():
        return [f"feature_{i}" for i in range(n_features)]
    df = pd.read_csv(file_path, nrows=0)
    cols_to_drop = [c for c in df.columns if "Unnamed" in c or "PERWT" in c]
    df = df.drop(columns=cols_to_drop)
    target_col = "UTILIZATION_reg"
    cols = [c for c in df.columns if c != target_col] + [target_col]
    names = [str(c) for c in cols]
    if len(names) != n_features:
        return [f"feature_{i}" for i in range(n_features)]
    return names


def _dedupe_names(names: Sequence[str]) -> List[str]:
    seen = {}
    out = []
    for raw_name in names:
        name = str(raw_name).strip() or "feature"
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count}")
    return out


class LLMJudge:
    def __init__(
        self,
        enabled: bool,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: str = ".cache/llm_cd/cache.json",
        sleep_seconds: float = 0.0,
    ):
        self.enabled = enabled
        self.api_key = api_key or os.getenv("LLMCD_API_KEY")
        self.base_url = base_url or os.getenv("LLMCD_BASE_URL") or "https://api.deepseek.com"
        self.model = model or os.getenv("LLMCD_MODEL") or "deepseek-v4-flash"
        self.sleep_seconds = sleep_seconds
        self.calls = 0
        self.cache_hits = 0
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, str] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

        self.client = None
        if self.enabled:
            if not self.api_key:
                raise ValueError("LLM is enabled but LLMCD_API_KEY is not set.")
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str) -> str:
        key = hashlib.sha256((self.model + "\n" + prompt).encode("utf-8")).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        if not self.enabled or self.client is None:
            raise RuntimeError("LLM judge is disabled.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            top_p=0.7,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        self.calls += 1
        self.cache[key] = content
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False))
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return content

    def score_independence(self, prompt: str) -> Tuple[float, float]:
        text = self.complete(prompt)
        nums = _extract_numbers(text)
        if not nums:
            return 0.5, 0.0
        score = float(nums[0])
        confidence = float(nums[1]) if len(nums) > 1 else 0.5
        if score > 1.0:
            score /= 100.0
        if confidence > 1.0:
            confidence /= 100.0
        return float(np.clip(score, 0.0, 1.0)), float(np.clip(confidence, 0.0, 1.0))

    def score_directed_edge(self, prompt: str) -> Tuple[str, float]:
        text = self.complete(prompt)
        scores = _extract_option_scores(text, ["KEEP", "FLIP", "REMOVE"])
        confidence = _extract_confidence(text)
        if not scores:
            return "KEEP", confidence
        return max(scores, key=scores.get), confidence

    def score_undirected_edge(self, prompt: str) -> Tuple[int, float]:
        text = self.complete(prompt)
        nums = _extract_numbers(text)
        if not nums:
            return 1, 0.0
        answer = int(round(float(nums[0])))
        confidence = float(nums[1]) if len(nums) > 1 else 0.5
        if confidence > 1.0:
            confidence /= 100.0
        return 1 if answer == 1 else 0, float(np.clip(confidence, 0.0, 1.0))

    def break_cycle(self, prompt: str, feature_names: Sequence[str]) -> Optional[Tuple[str, int, int]]:
        text = self.complete(prompt)
        action = "remove" if "remove" in text.lower() else "reverse" if "reverse" in text.lower() else None
        if action is None:
            return None
        found = [i for i, name in enumerate(feature_names) if name in text]
        if len(found) >= 2:
            return action, found[0], found[1]
        return None


def discover_graph(
    data: np.ndarray,
    v: np.ndarray,
    target_idx: int,
    feature_names: List[str],
    feature_descriptions: Dict[str, str],
    alpha: float = 0.05,
    indep_test: str = "auto",
    use_llm: bool = False,
    ci_threshold: float = 0.001,
    max_llm_pairs: int = 40,
    llm_judge: Optional[LLMJudge] = None,
    random_state: int = 42,
    show_progress: bool = False,
) -> GraphDiscoveryResult:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT

    rng = np.random.default_rng(random_state)
    data = np.asarray(data, dtype=float)
    n_features = data.shape[1]
    test_name = _choose_independence_test(v, indep_test)
    pc_data = data
    try:
        cg = pc(pc_data, alpha=alpha, indep_test=test_name, stable=True, uc_rule=0, uc_priority=2, show_progress=show_progress)
    except ValueError as exc:
        if "singular" not in str(exc).lower() or test_name not in {"fisherz", "kci"}:
            raise
        pc_data = data + rng.normal(0.0, 1e-6, size=data.shape)
        cg = pc(pc_data, alpha=alpha, indep_test=test_name, stable=True, uc_rule=0, uc_priority=2, show_progress=show_progress)

    graph = np.zeros((n_features, n_features), dtype=int)
    directed_edges = [(int(i), int(j)) for i, j in cg.find_fully_directed()]
    undirected_edges = sorted({tuple(sorted((int(i), int(j)))) for i, j in cg.find_undirected()})

    llm = llm_judge or LLMJudge(enabled=False)

    # LLM-CD skeleton-stage approximation: ask LLM for marginal CI cases near alpha.
    # The official code asks inside PC for every near-threshold conditional test.
    # Here we keep the same role for the LLM without patching causallearn internals.
    if use_llm and ci_threshold > 0 and max_llm_pairs > 0:
        cit = CIT(pc_data, test_name)
        candidates = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    p_value = float(cit(i, j, ()))
                except Exception:
                    continue
                delta = abs(p_value - alpha)
                if delta <= ci_threshold:
                    candidates.append((delta, i, j))
        candidates.sort(key=lambda item: item[0])
        for _, i, j in candidates[:max_llm_pairs]:
            prompt = conditional_independence_prompt(i, j, [], feature_names, feature_descriptions)
            score, _ = llm.score_independence(prompt)
            has_edge = cg.G.graph[i, j] != 0 or cg.G.graph[j, i] != 0
            if score >= 0.5 and has_edge:
                directed_edges = [edge for edge in directed_edges if set(edge) != {i, j}]
                undirected_edges = [edge for edge in undirected_edges if set(edge) != {i, j}]
            elif score < 0.5 and not has_edge:
                undirected_edges.append((i, j))

    for i, j in directed_edges:
        action = "KEEP"
        if use_llm:
            prompt = directed_edge_prompt(i, j, feature_names, feature_descriptions)
            action, _ = llm.score_directed_edge(prompt)
        if action == "KEEP":
            graph[i, j] = 1
        elif action == "FLIP":
            graph[j, i] = 1
        elif action == "REMOVE":
            continue

    for i, j in sorted(set(undirected_edges)):
        if use_llm:
            prompt = undirected_edge_prompt(i, j, feature_names, feature_descriptions)
            direction, _ = llm.score_undirected_edge(prompt)
            if direction == 1:
                graph[i, j] = 1
            else:
                graph[j, i] = 1
        else:
            # Deterministic fallback for dry runs. Prefer direct target parents so
            # parent-only prediction can run before an API key is configured.
            if j == target_idx:
                graph[i, j] = 1
            elif i == target_idx:
                graph[j, i] = 1
            else:
                graph[min(i, j), max(i, j)] = 1

    graph = _break_cycles(graph, data, feature_names, feature_descriptions, llm if use_llm else None)
    parents = sorted(np.where(graph[:, target_idx] == 1)[0].astype(int).tolist())
    if not parents:
        parents = _fallback_target_parents(data, target_idx, k=min(5, max(1, n_features - 1)))
        for parent in parents:
            graph[parent, target_idx] = 1

    return GraphDiscoveryResult(
        parents=parents,
        graph=graph,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        pc_directed_edges=directed_edges,
        pc_undirected_edges=undirected_edges,
        llm_calls=llm.calls,
        llm_cache_hits=llm.cache_hits,
    )


def run_llm_cd_baseline(
    dataset: str,
    use_llm: bool = False,
    alpha: float = 0.05,
    indep_test: str = "auto",
    sample_size: Optional[int] = 1000,
    predictor: str = "rf",
    random_state: int = 42,
    ci_threshold: float = 0.001,
    max_llm_pairs: int = 40,
    cache_path: str = ".cache/llm_cd/cache.json",
) -> BaselineResult:
    full, y, v, num_classes_dict, target_idx = load_baseline_arrays(dataset)
    feature_names, descriptions = feature_metadata(dataset, full.shape[1])

    rng = np.random.default_rng(random_state)
    if sample_size is not None and sample_size > 0 and len(full) > sample_size:
        graph_indices = rng.choice(len(full), size=sample_size, replace=False)
        graph_data = full[graph_indices]
    else:
        graph_data = full

    llm = LLMJudge(enabled=use_llm, cache_path=cache_path)
    discovery = discover_graph(
        graph_data,
        v=v,
        target_idx=target_idx,
        feature_names=feature_names,
        feature_descriptions=descriptions,
        alpha=alpha,
        indep_test=indep_test,
        use_llm=use_llm,
        ci_threshold=ci_threshold,
        max_llm_pairs=max_llm_pairs,
        llm_judge=llm,
        random_state=random_state,
    )

    task_type = "regression" if v[target_idx] == 0 else "classification"
    metrics = evaluate_parent_predictor(
        full,
        y,
        target_idx,
        discovery.parents,
        task_type=task_type,
        predictor=predictor,
        random_state=random_state,
    )
    return BaselineResult(
        dataset=dataset,
        task_type=task_type,
        target_idx=target_idx,
        target_name=feature_names[target_idx],
        parents=discovery.parents,
        parent_names=[feature_names[idx] for idx in discovery.parents],
        graph=discovery.graph,
        metrics=metrics,
        llm_calls=discovery.llm_calls,
        llm_cache_hits=discovery.llm_cache_hits,
    )


def evaluate_parent_predictor(
    data: np.ndarray,
    y: np.ndarray,
    target_idx: int,
    parents: Sequence[int],
    task_type: str,
    predictor: str,
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    feature_indices = [idx for idx in parents if idx != target_idx]
    if not feature_indices:
        feature_indices = _fallback_target_parents(data, target_idx, k=1)
    x = data[:, feature_indices]
    y = y.astype(float)

    stratify = y if task_type == "classification" and len(np.unique(y)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )
    if task_type == "classification":
        y_train_int = y_train.astype(int)
        y_test_int = y_test.astype(int)
        model = _make_classifier(predictor, random_state)
        model.fit(x_train, y_train_int)
        pred = model.predict(x_test).astype(int)
        metrics = _classification_metrics(y_test_int, pred, model, x_test)
    else:
        model = _make_regressor(predictor, random_state)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics = _regression_metrics(y_test, pred)
    return {"test": metrics}


def _make_classifier(predictor: str, random_state: int):
    if predictor == "logistic":
        return LogisticRegression(max_iter=4000, class_weight="balanced")
    if predictor == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=random_state)
    return RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=random_state)


def _make_regressor(predictor: str, random_state: int):
    if predictor == "linear":
        return LinearRegression()
    if predictor == "mlp":
        return MLPRegressor(hidden_layer_sizes=(128,), max_iter=500, random_state=random_state)
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)


def _classification_metrics(y_true, y_pred, model, x_test) -> Dict[str, float]:
    average = "binary" if len(np.unique(y_true)) <= 2 else "weighted"
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    auc = 0.0
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_test)
            if probs.shape[1] == 2:
                auc = roc_auc_score(y_true, probs[:, 1])
            else:
                auc = roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(auc),
    }


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _choose_independence_test(v: np.ndarray, indep_test: str) -> str:
    if indep_test != "auto":
        return indep_test
    return "chisq" if np.all(v == 1) else "fisherz"


def _fallback_target_parents(data: np.ndarray, target_idx: int, k: int = 5) -> List[int]:
    target = data[:, target_idx]
    scores = []
    for idx in range(data.shape[1]):
        if idx == target_idx:
            continue
        try:
            score = abs(np.corrcoef(data[:, idx], target)[0, 1])
            if np.isnan(score):
                score = 0.0
        except Exception:
            score = 0.0
        scores.append((score, idx))
    scores.sort(reverse=True)
    return sorted(idx for _, idx in scores[:k])


def _break_cycles(
    graph: np.ndarray,
    data: np.ndarray,
    feature_names: Sequence[str],
    feature_descriptions: Dict[str, str],
    llm: Optional[LLMJudge],
) -> np.ndarray:
    import networkx as nx

    graph = graph.copy()
    for _ in range(graph.shape[0] * graph.shape[0]):
        digraph = nx.DiGraph()
        digraph.add_nodes_from(range(graph.shape[0]))
        digraph.add_edges_from([(i, j) for i, j in zip(*np.where(graph == 1))])
        try:
            cycle = nx.find_cycle(digraph, orientation="original")
        except nx.NetworkXNoCycle:
            return graph
        cycle_nodes = [edge[0] for edge in cycle]
        cycle_edges = [(edge[0], edge[1]) for edge in cycle]
        action = None
        if llm is not None:
            prompt = cycle_break_prompt(cycle_nodes, list(feature_names), feature_descriptions)
            action = llm.break_cycle(prompt, feature_names)
        if action is not None:
            op, src, dst = action
            if graph[src, dst] == 1:
                if op == "reverse":
                    graph[src, dst] = 0
                    graph[dst, src] = 1
                else:
                    graph[src, dst] = 0
                continue
        weakest = min(cycle_edges, key=lambda edge: _abs_corr(data[:, edge[0]], data[:, edge[1]]))
        graph[weakest[0], weakest[1]] = 0
    return graph


def _abs_corr(a, b) -> float:
    try:
        val = abs(np.corrcoef(a, b)[0, 1])
        return 0.0 if np.isnan(val) else float(val)
    except Exception:
        return 0.0


def _extract_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text or "")]


def _extract_option_scores(text: str, options: Iterable[str]) -> Dict[str, float]:
    scores = {}
    for option in options:
        match = re.search(rf'"?{re.escape(option)}"?\s*:\s*([-+]?\d*\.?\d+)', text or "", flags=re.I)
        if match:
            value = float(match.group(1))
            scores[option] = value / 100.0 if value > 1.0 else value
    return scores


def _extract_confidence(text: str) -> float:
    nums = _extract_numbers(text)
    if not nums:
        return 0.5
    value = nums[-1]
    if value > 1.0:
        value /= 100.0
    return float(np.clip(value, 0.0, 1.0))


def result_to_jsonable(result: BaselineResult) -> Dict:
    return {
        "dataset": result.dataset,
        "task_type": result.task_type,
        "target_idx": result.target_idx,
        "target_name": result.target_name,
        "parents": result.parents,
        "parent_names": result.parent_names,
        "metrics": result.metrics,
        "llm_calls": result.llm_calls,
        "llm_cache_hits": result.llm_cache_hits,
        "graph": result.graph.tolist(),
    }


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="LLM-CD style parent-only prediction baseline")
    parser.add_argument("--dataset", default="creditcard", choices=sorted(DATASET_LOADERS))
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM calls for uncertain CI and edge orientation")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--indep_test", default="auto", choices=["auto", "fisherz", "chisq", "gsq", "kci"])
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--predictor", default="rf", choices=["rf", "logistic", "linear", "mlp"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--ci_threshold", type=float, default=0.001)
    parser.add_argument("--max_llm_pairs", type=int, default=40)
    parser.add_argument("--cache_path", default=".cache/llm_cd/cache.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    result = run_llm_cd_baseline(
        dataset=args.dataset,
        use_llm=args.use_llm,
        alpha=args.alpha,
        indep_test=args.indep_test,
        sample_size=args.sample_size,
        predictor=args.predictor,
        random_state=args.random_state,
        ci_threshold=args.ci_threshold,
        max_llm_pairs=args.max_llm_pairs,
        cache_path=args.cache_path,
    )
    payload = result_to_jsonable(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
