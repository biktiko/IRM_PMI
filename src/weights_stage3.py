import pandas as pd
import numpy as np
from src.common import norm

STAGE3_WEIGHTS_DATA = [
    {"section": "First Contact", "section_total_weight": 0.1, "question_key": "SE-ն բարեհամբյուր կերպով ողջունել և ներկայացել է Ձեզ։", "weight_in_section": 0.4, "weight_in_scenario": 0.04},
    {"section": "First Contact", "section_total_weight": 0.1, "question_key": "Ձեր ժամանման պահին խանութը մաքո՞ւր էր։", "weight_in_section": 0.2, "weight_in_scenario": 0.02},
    {"section": "First Contact", "section_total_weight": 0.1, "question_key": "SE-ի ընդհանուր արտաքինն ու մաքրությունը տեղի՞ն էր։", "weight_in_section": 0.2, "weight_in_scenario": 0.02},
    {"section": "First Contact", "section_total_weight": 0.1, "question_key": "SE-ն կրում էր անվանապիտակ", "weight_in_section": 0.2, "weight_in_scenario": 0.02},
    
    {"section": "Identification", "section_total_weight": 0.25, "question_key": "Մինչ խոսակցությունը սկսելը հարցրե՞լ է արդյոք հանդիսանում եք Չափահաս Ծխող", "weight_in_section": 0.4, "weight_in_scenario": 0.1},
    {"section": "Identification", "section_total_weight": 0.25, "question_key": "Մինչ խոսակցությունը սկսելը հարցրե՞լ է արդյոք հանդիսանում եք ծխախոտային կամ նիկոտինային արտադրանք օգտագործող", "weight_in_section": 0.4, "weight_in_scenario": 0.1},
    {"section": "Identification", "section_total_weight": 0.25, "question_key": "Արդյոք խնդրել է ներկայացնել անձը հաստատող փաստաթուղթ", "weight_in_section": 0.2, "weight_in_scenario": 0.05},

    {"section": "Assessment", "section_total_weight": 0.3, "question_key": "SE-ն հետաքրքրվել է, թե հաճախորդն ինչ գիտի IQOS սարքերի (ներառյալ IQOS ILUMA կամ IQOS ILUMA i սարքերի մասին)", "weight_in_section": 0.1, "weight_in_scenario": 0.03},
    {"section": "Assessment", "section_total_weight": 0.3, "question_key": "SE-ն տվել է ձեզ հարցեր, որպեսզի հասկանա ինչ կարիքներ, նախասիրություններ կամ ցավոտ կետեր ունեք։", "weight_in_section": 0.1, "weight_in_scenario": 0.03},
    
    {"section": "Additional Benefits/CATM", "section_total_weight": 0.3, "question_key": "SE-ն կիսվել է ձեզ հետ գոնե որևէ մի CATM հաղորդագրությամբ", "weight_in_section": 0.07, "weight_in_scenario": 0.021},

    {"section": "Product Presentation", "section_total_weight": 0.3, "question_key": "SE-ն նշել է առկա բոլոր սարքերի մոդելները (IQOS ILUMA (i)PRIME, IQOS ILUMA (i), IQOS ILUMA (i) ONE )", "weight_in_section": 0.04, "weight_in_scenario": 0.012},
    {"section": "Product Presentation", "section_total_weight": 0.3, "question_key": "SE-ն խոսել է սարքերի մասին՝ պահպանելով հիերարխիան (IQOS ILUMA (i)PRIME, IQOS ILUMA (i), IQOS ILUMA (i) ONE )", "weight_in_section": 0.04, "weight_in_scenario": 0.012},
    {"section": "Product Presentation", "section_total_weight": 0.3, "question_key": "SE-ն նշել է՝ ինչ ֆունկցիաներ ունեն սարքերը", "weight_in_section": 0.07, "weight_in_scenario": 0.021},
    {"section": "Product Presentation", "section_total_weight": 0.3, "question_key": "SE-ն նշել է SMARTCORE ինդուկցիոն համակարգի մասին", "weight_in_section": 0.1, "weight_in_scenario": 0.03},
    {"section": "Product Presentation", "section_total_weight": 0.3, "question_key": "SE-ն առաջարկել է դուրս գալ վաճառակետից և փորձարկել սարքը", "weight_in_section": 0.1, "weight_in_scenario": 0.03},
    {"section": "Product Presentation", "section_total_weight": 0.3, "question_key": "SE-ն ներկայացրել է, որ IQOS ILUMA սարքերը չունեն լեզվակ, մաքրման կարիք չունեն", "weight_in_section": 0.05, "weight_in_scenario": 0.015},
    {"section": "Product Presentation", "section_total_weight": 0.3, "question_key": "SE-ն նշել է, որ IQOS ILUMA-ն համատեղելի է միայն TEREA SMARTCORE սթիքերի հետ։", "weight_in_section": 0.06, "weight_in_scenario": 0.018},

    {"section": "Taste Presentation", "section_total_weight": 0.3, "question_key": "Ելնելով Ձեր համային նախասիրություններից SE-ն խորհուրդ է տվել, թե TEREA-ի որ տարբերակը կարող եք գնել։", "weight_in_section": 0.08, "weight_in_scenario": 0.024},
    {"section": "Taste Presentation", "section_total_weight": 0.3, "question_key": "Նկարագրել է արդյոք TEREA սթիքերի առկա տեսականին", "weight_in_section": 0.07, "weight_in_scenario": 0.021},
    {"section": "Taste Presentation", "section_total_weight": 0.3, "question_key": "Առաջարկել է արդյոք փորձել 2+ համ", "weight_in_section": 0.06, "weight_in_scenario": 0.018},
    {"section": "Taste Presentation", "section_total_weight": 0.3, "question_key": "Ներկայացրել է նոր համերն ու կապսուլները և ցույց տվեց ինչպես օգտագործել", "weight_in_section": 0.06, "weight_in_scenario": 0.018},

    {"section": "CO meter", "section_total_weight": 0.05, "question_key": "SE-ն պատմել է СO մետրի մասին", "weight_in_section": 0.3, "weight_in_scenario": 0.015},
    {"section": "CO meter", "section_total_weight": 0.05, "question_key": "SE-ն առաջարկել է ձեզ անցնել CO meter թեսթը", "weight_in_section": 0.25, "weight_in_scenario": 0.0125},
    {"section": "CO meter", "section_total_weight": 0.05, "question_key": "SE-ն բացատրել է թեսթի արդյունքները", "weight_in_section": 0.25, "weight_in_scenario": 0.0125},
    {"section": "CO meter", "section_total_weight": 0.05, "question_key": "SE-ն տեղեկացրե՞լ է, որ 1 շաբաթ միայն IQOS Iluma օգտագործելու դեպքում ցուցանիշը բարելավվելու է", "weight_in_section": 0.2, "weight_in_scenario": 0.01},

    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն հետաքրքրվել է ինչու չեք գնում սարքը", "weight_in_section": 0.3, "weight_in_scenario": 0.06},
    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն տեղեկացրել է, որ ավելի լավ հագեցվածության զգացումի համար հարկավոր է բավարար դադարներ(առնվազն 10 վյրկ) տալ ներքաշումների միջև, որպեսզի սթիքը հասցնի տաքանալ։", "weight_in_section": 0.2, "weight_in_scenario": 0.04},

    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն տեղեկացրել է, որ հարկավոր է փորձել 1-ից ավել համ՝ գտնելու այն համը որն առավել հագեցվածություն է բերում։", "weight_in_section": 0.25, "weight_in_scenario": 0.05},
    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն խորհուրդ է տվել համատեղել մի քանի համ, ավելի լավ հագեցվածության համար։", "weight_in_section": 0.0, "weight_in_scenario": 0.0},
    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն տեղեկացրել է , որ ադապտացիայի և ավելի լավ հագեցվածության համար հարկավոր է օգտագործել միայն սարքը ՝առնվազն 2 շաբաթ, որպեսզի ընտելանան IQOS-ի փորձին", "weight_in_section": 0.0, "weight_in_scenario": 0.0},
    
    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն տեղեկացրել է, որ IQOS ILUMA-ն ապահովում է սիգարետին նման հագեցվածություն:", "weight_in_section": 0.25, "weight_in_scenario": 0.05},
    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն տեղեկացրել է, որն ILUMA-ն ապահովում է իրական ծխախոտի համ և հագեցվածություն` սպառողներին տրամադրելով այլ առավելություններ ևս, որպեսզի այն դրականորեն տարբերակվի սիգարետներից։", "weight_in_section": 0.0, "weight_in_scenario": 0.0},
    {"section": "Objections Handling", "section_total_weight": 0.2, "question_key": "SE-ն տեղեկացրել է ,որ մեր պնդումները հիմնված են ՖՄԻ կողմից կատարված բազմաթիվ հետազատությունների վրա", "weight_in_section": 0.0, "weight_in_scenario": 0.0},

    {"section": "Farewell", "section_total_weight": 0.1, "question_key": "SE-ն առաջարկել է ցանկացած հարցի դեպքում կրկին վաճառակետ այցելել կամ զանգահարել IQOS սպասարկման կենտրոն։ (web)", "weight_in_section": 1.0, "weight_in_scenario": 0.1},
]

def get_static_weights() -> pd.DataFrame:
    df = pd.DataFrame(STAGE3_WEIGHTS_DATA)
    df["scenario"] = "Stage3"
    df["qkey_norm"] = df["question_key"].apply(norm)
    return df

def apply_stage3_rules(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    # Rule 1: CATM (S to AB) -> R
    catm_q_norm = norm("SE-ն կիսվել է ձեզ հետ գոնե որևէ մի CATM հաղորդագրությամբ")
    catm_subs_norm = [
        norm("Ամբողջովին IQOS ILUMA-ին անցումը ավելի քիչ վնասակար է, քան ծխել շարունակելը։"),
        norm("Ամբողջովին IQOS ILUMA-ին անցումը ավելի քիչ ռիսկ է ներկյացնում Ձեր առողջությանը, քան ծխել շարունակելը։"),
        norm("IQOS ILUMA-ի օգտագործումն ավելի քիչ վտանգ է ներկայացնում Ձեր առողջության համար, քան ծխելը շարունակելը:"),
        norm("IQOS ILUMA-ն ավելի քիչ հոտ է առաջացնում օգտագործման ժամանակ՝ IQOS-ի նախորդ սերունդների համեմատ:*"),
        norm("IQOS ILUMA-ն զգալիորեն ավելի քիչ հետքեր է թողնում Ձեր ատամների վրա,* քան սիգարետը:"),
        norm("IQOS ILUMA-ին անցում կատարած ծխողներն ասում են, որ իրենց շնչառությունն ավելի թարմ է, քան սիգարետ ծխելու ժամանակ:"),
        norm("IQOS ILUMA-ին անցում կատարած ծխողներն ասում են, որ իրենց բերանում մնացող զգացողությունն ու համը պակաս տհաճ են, քան սիգարետ ծխելուց հետո:"),
        norm("IQOS ILUMA-ն բացասական ազդեցություն չի ունենում փակ տարածքներում օդի որակի վրա։"),
        norm("IQOS ILUMA-ն երկրորդային ծուխ չի առաջացնում:"),
        norm("IQOS ILUMA-ն ապահովում է ավելի հաճելի փորձ՝ ի համեմատ IQOS-ի նախորդ սերունդների հետ։* Շատ սպառողներ համաձայն են, որ IQOS ILUMA-ն ավելի լավ համ է ապահովում, քան IQOS-ի նախորդ սերունդները:")
    ]

    # Rule 2 Group 1:
    grp1_main_norm = norm("SE-ն տեղեկացրել է, որ հարկավոր է փորձել 1-ից ավել համ՝ գտնելու այն համը որն առավել հագեցվածություն է բերում։")
    grp1_subs_norm = [
        norm("SE-ն խորհուրդ է տվել համատեղել մի քանի համ, ավելի լավ հագեցվածության համար։"),
        norm("SE-ն տեղեկացրել է , որ ադապտացիայի և ավելի լավ հագեցվածության համար հարկավոր է օգտագործել միայն սարքը ՝առնվազն 2 շաբաթ, որպեսզի ընտելանան IQOS-ի փորձին")
    ]
    
    # Rule 2 Group 2:
    grp2_main_norm = norm("SE-ն տեղեկացրել է, որ IQOS ILUMA-ն ապահովում է սիգարետին նման հագեցվածություն:")
    grp2_subs_norm = [
        norm("SE-ն տեղեկացրել է, որն ILUMA-ն ապահովում է իրական ծխախոտի համ և հագեցվածություն` սպառողներին տրամադրելով այլ առավելություններ ևս, որպեսզի այն դրականորեն տարբերակվի սիգարետներից։"),
        norm("SE-ն տեղեկացրել է ,որ մեր պնդումները հիմնված են ՖՄԻ կողմից կատարված բազմաթիվ հետազատությունների վրա")
    ]

    for store in long_df["store"].unique():
        store_mask = long_df["store"] == store
        
        # Check CATM
        catm_sub_mask = store_mask & long_df["qkey_norm"].isin(catm_subs_norm) & (long_df["answer_bin"] == 1)
        if catm_sub_mask.any():
            long_df.loc[store_mask & (long_df["qkey_norm"] == catm_q_norm), "answer_bin"] = 1
            
        # Check Grp1
        grp1_sub_mask = store_mask & long_df["qkey_norm"].isin(grp1_subs_norm) & (long_df["answer_bin"] == 1)
        if grp1_sub_mask.any():
            long_df.loc[store_mask & (long_df["qkey_norm"] == grp1_main_norm), "answer_bin"] = 1

        # Check Grp2
        grp2_sub_mask = store_mask & long_df["qkey_norm"].isin(grp2_subs_norm) & (long_df["answer_bin"] == 1)
        if grp2_sub_mask.any():
            long_df.loc[store_mask & (long_df["qkey_norm"] == grp2_main_norm), "answer_bin"] = 1

    return long_df
