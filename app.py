import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv("data/clean/cleaned_data.csv") 
    return df

df = load_data()

st.title("Dashboard d'Analyse Exploratoire")

tab0, tab1, tab2, tab3, tab4, tab5= st.tabs(["Introduction","tendance global", "Impact du Développement Économique sur la Santé", "Impact des Infrastructures Médicales","Influence de la Population et de la Densité Urbaine","Étude de Cas"])
with tab0:
    st.markdown(
        """
        <h1 style="text-align: center; color: #4CAF50;">📌 Introduction</h1>
        <p style="text-align: justify; font-size: 18px;">
        La santé et le développement économique sont étroitement liés. L’espérance de vie et la mortalité infantile 
        reflètent les conditions sanitaires et le niveau de vie d’un pays. Cette étude explore comment les facteurs 
        socio-économiques influencent ces indicateurs, en analysant le rôle du PIB par habitant, des dépenses de santé, 
        des infrastructures médicales et de la densité de population sur la qualité de vie à l’échelle mondiale.

        Source de nos données : Our World Data.<br>
        type de données :<br>
        Mortalité infantille : decés pour 1000 niassance <br>
        Espérance de vie : Age moyen d'une personne <br>
        Dépenses de santé : pourcentage du PIB (USD)<br>
        PIB par habitant : USD<br>
        Nombre de lits d'hôpitaux : nombre de lits pour 1000 habitants<br>
        Densité de population : nombre d'habitants par km²<br>
        population : nombre d'habitants
        </p>
        """, unsafe_allow_html=True
    )
with tab1:
    global_trend = df.groupby('Year')['Child Mortality'].mean()
    fig, ax = plt.subplots()
    ax.plot(global_trend.index, global_trend.values, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Année')
    ax.set_ylabel('Mortalité Infantile (pour 1000 naissances)')
    ax.set_title('Évolution de la Mortalité Infantile dans le Monde')
    ax.grid()
    st.pyplot(fig)

    global_trend = df.groupby('Year')['Life Expectancy'].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 5)) 
    ax2.plot(global_trend.index, global_trend.values, marker='o', linestyle='-', color='g')
    ax2.set_xlabel('Année')
    ax2.set_ylabel('Espérance de vie')
    ax2.set_title('Évolution de l\'espérance de vie dans le Monde')
    ax2.grid()
    st.pyplot(fig2)

    global_mortality_trend = df.groupby('Year')['Child Mortality'].mean()
    global_life_expectancy_trend = df.groupby('Year')['Life Expectancy'].mean()

    fig2, ax2 =plt.subplots(figsize=(10, 5))
    ax2.plot(global_mortality_trend.index, global_mortality_trend.values, marker='o', linestyle='-', color='b', label='Mortalité Infantile')
    ax2.plot(global_life_expectancy_trend.index, global_life_expectancy_trend.values, marker='s', linestyle='-', color='g', label='Espérance de Vie')
    ax2.set_xlabel('Année')
    ax2.set_ylabel('Valeurs')
    ax2.set_title('Évolution de la Mortalité Infantile et de lEspérance de Vie')
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

    st.subheader("Régression linéaire entre Mortalité Infantile et Espérance de Vie")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=df['Child Mortality'], y=df['Life Expectancy'], scatter_kws={'alpha':0.8}, line_kws={'color':'red'}, ax=ax)
    ax.set_xlabel('Mortalité Infantile (pour 1000 naissances)')
    ax.set_ylabel('Espérance de Vie (années)')
    ax.set_title('Régression linéaire entre Mortalité Infantile et Espérance de Vie')
    ax.grid()
    st.pyplot(fig)

    continent_map = {
    "Africa": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde", "Central African Republic", 
               "Chad", "Comoros", "Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", 
               "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", 
               "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", 
               "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", 
               "Uganda", "Zambia", "Zimbabwe"],

    "Asia": ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China", "Georgia", "India", 
             "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", 
             "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia", 
             "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Timor-Leste", "Turkey", "Turkmenistan", "United Arab Emirates", 
             "Uzbekistan", "Vietnam", "Yemen"],

    "Europe": ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", 
               "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Liechtenstein", 
               "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", 
               "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", 
               "Vatican City"],

    "North America": ["Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", 
                      "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia", 
                      "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States"],

    "South America": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"],

    "Oceania": ["Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa", 
                "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"]
    }
    def get_continent(country):
        for continent, countries in continent_map.items():
            if country in countries:
                return continent
        return "Other" 
    df["Continent"] = df["Country"].apply(get_continent)
    df = df[df["Continent"] != "Other"]
    df_continent = df.groupby(["Year", "Continent"])["Life Expectancy"].mean().reset_index()
    st.subheader("Évolution de l'espérance de vie dans le temps par continent")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_continent, x="Year", y="Life Expectancy", hue="Continent", marker="o", ax=ax)
    ax.set_title("Évolution de l'espérance de vie dans le temps par continent")
    ax.set_xlabel("Année")
    ax.set_ylabel("Espérance de Vie Moyenne")
    ax.legend(title="Continent")
    st.pyplot(fig)

    st.subheader("Top 10 des pays avec la meilleure espérance de vie (moyenne)")
    top_countries = df.groupby('Country')['Life Expectancy'].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_countries.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel('Espérance de Vie Moyenne (années)')
    ax.set_ylabel('Pays')
    ax.set_title('Top 10 des pays avec la meilleure espérance de vie (moyenne)')
    ax.grid()
    st.pyplot(fig)

    st.subheader("Top 10 des pays avec la meilleure dépense santé (moyenne)")
    top_countries = df.groupby('Country')['Health Expenditure'].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_countries.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel("Dépense de Santé Moyenne")
    ax.set_ylabel("Pays")
    ax.set_title("Top 10 des pays avec la meilleure dépense santé (moyenne)")
    ax.grid()
    st.pyplot(fig)

with tab2:
    st.subheader("Lien entre PIB par habitant et espérance de vie")
    fig, ax = plt.subplots(figsize=(17, 6))
    sns.scatterplot(data=df, x="GDP per Capita", y="Life Expectancy", alpha=0.5, ax=ax)
    ax.set_title("Lien entre PIB par habitant et espérance de vie")
    ax.grid(axis='y', linewidth=0.5)
    ax.set_xlabel("PIB par habitant")
    ax.set_ylabel("Espérance de Vie")
    st.pyplot(fig)

    st.subheader("Évolution de l'espérance de vie dans les 10 pays les plus riches")
    top_10_countries = df.groupby("Country")["GDP per Capita"].mean().nlargest(10).index
    df_top_10_time = df[df["Country"].isin(top_10_countries)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_top_10_time, x="Year", y="Life Expectancy", hue="Country", marker="o", palette="tab10", ax=ax)
    ax.set_title("Évolution de l'espérance de vie dans les 10 pays les plus riches")
    ax.set_xlabel("Année")
    ax.set_ylabel("Espérance de Vie Moyenne")
    ax.legend(title="Pays", bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    st.subheader("Évolution de l'espérance de vie dans les 10 pays les plus pauvres")
    top_10_poor_countries = df.groupby("Country")["GDP per Capita"].mean().nsmallest(10).index
    df_top_10_poor_time = df[df["Country"].isin(top_10_poor_countries)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_top_10_poor_time, x="Year", y="Life Expectancy", hue="Country", marker="o", palette="tab10", ax=ax)
    ax.set_title("Évolution de l'espérance de vie dans les 10 pays les plus pauvres")
    ax.set_xlabel("Année")
    ax.set_ylabel("Espérance de Vie Moyenne")
    ax.legend(title="Pays", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    st.subheader("Impact de la richesse sur la mortalité infantile et l'espérance de vie")
    df["GDP Category"] = pd.qcut(df["GDP per Capita"], q=10, labels=[f"Décile {i+1}" for i in range(10)])
    df_gdp_analysis = df.groupby("GDP Category").agg({
        "Child Mortality": "mean",
        "Life Expectancy": "mean"
    }).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_gdp_analysis, x="GDP Category", y="Child Mortality", palette="Blues_r", ax=ax1)
    ax1.set_ylabel("Mortalité Infantile Moyenne", color="blue")
    ax2 = ax1.twinx()
    sns.lineplot(data=df_gdp_analysis, x="GDP Category", y="Life Expectancy", marker="o", color="red", linewidth=2, ax=ax2)
    ax2.set_ylabel("Espérance de Vie Moyenne", color="red")
    ax1.set_xlabel("Catégorie de PIB par habitant (en déciles)")
    ax1.set_title("Impact de la richesse sur la mortalité infantile et l'espérance de vie")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.subheader("Corrélation entre Mortalité Infantile, Espérance de Vie et PIB par Habitant")
    df_filtered = df[['Child Mortality', 'Life Expectancy', 'GDP per Capita']].dropna()
    correlation_matrix = df_filtered.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Corrélation entre Mortalité Infantile, Espérance de Vie et PIB par Habitant")
    st.pyplot(fig)

with tab3:
    st.subheader("Relation entre les dépenses de santé et l'espérance de vie")
    df_filtered = df[['Health Expenditure', 'Life Expectancy']].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df_filtered['Health Expenditure'], y=df_filtered['Life Expectancy'], scatter_kws={'alpha': 0.5}, line_kws={"color": "red"}, ax=ax)
    ax.set_xlabel("Dépenses de santé")
    ax.set_ylabel("Espérance de vie (années)")
    ax.set_title("Relation entre les dépenses de santé et l'espérance de vie")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Boîtes à moustaches : Espérance de vie par continent")
    if {'Continent', 'Life Expectancy'}.issubset(df_continent.columns):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_continent, x="Continent", y="Life Expectancy", palette="Set2", ax=ax)
        ax.set_xlabel("Continent")
        ax.set_ylabel("Espérance de vie (années)")
        ax.set_title("Boîtes à moustaches : Espérance de vie par continent")
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("Top 10 des pays avec les plus grosses dépenses de santé (2000-2023, avec USA)")
    if {'Health Expenditure', 'Country', 'Year'}.issubset(df.columns):
        df_filtered = df[(df["Year"] >= 2000) & (df["Year"] <= 2023)]
        top_10_countries = df_filtered.groupby("Country")["Health Expenditure"].mean().nlargest(10).reset_index()
        if "United States" not in top_10_countries["Country"].values:
            us_expenditure = df_filtered[df_filtered["Country"] == "United States"]["Health Expenditure"].mean()
            if not pd.isna(us_expenditure): 
                us_row = pd.DataFrame({"Country": ["United States"], "Health Expenditure": [us_expenditure]})
                top_10_countries = pd.concat([top_10_countries, us_row]).sort_values(by="Health Expenditure", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=top_10_countries, x="Health Expenditure", y="Country", palette="Blues_r", ax=ax)
        for index, value in enumerate(top_10_countries["Health Expenditure"]):
            ax.text(value + 0.1, index, f"{value:.2f}", va="center")
        ax.set_xlabel("Dépenses de santé moyennes % du PIB")
        ax.set_ylabel("Pays")
        ax.set_title("Top 10 des pays avec les plus grosses dépenses de santé (2000-2023, avec USA)")
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("Top 10 des pays avec les plus grosses dépenses de santé (1960-2023, avec USA)")
    if {'Health Expenditure', 'Country', 'Year'}.issubset(df.columns):
        df_filtered = df[(df["Year"] >= 1960) & (df["Year"] <= 2023)]
        top_10_countries = df_filtered.groupby("Country")["Health Expenditure"].mean().nlargest(10).reset_index()
        if "United States" not in top_10_countries["Country"].values:
            us_expenditure = df_filtered[df_filtered["Country"] == "United States"]["Health Expenditure"].mean()
            if not pd.isna(us_expenditure): 
                us_row = pd.DataFrame({"Country": ["United States"], "Health Expenditure": [us_expenditure]})
                top_10_countries = pd.concat([top_10_countries, us_row]).sort_values(by="Health Expenditure", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=top_10_countries, x="Health Expenditure", y="Country", palette="Blues_r", ax=ax)
        for index, value in enumerate(top_10_countries["Health Expenditure"]):
            ax.text(value + 0.1, index, f"{value:.2f}", va="center")
        ax.set_xlabel("Dépenses de santé moyennes % du PIB")
        ax.set_ylabel("Pays")
        ax.set_title("Top 10 des pays avec les plus grosses dépenses de santé (1960-2023, avec USA)")
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("Dépenses de santé par année")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x='Year', y='Health Expenditure', data=df, ax=ax)
    ax.set_title("Dépenses de santé par année")
    ax.set_xlabel("Année")
    ax.set_ylabel("Dépenses de santé")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

    st.subheader("Mortalité infantile moyenne par nombre de lits d'hôpitaux")
    df['Hospital Beds Category'] = pd.cut(df['Hospital Beds'],
                                        bins=[0, 1, 2, 3, 5, float('inf')],
                                        labels=['0', '1', '2', '3', '4'])
    grouped_df = df.groupby('Hospital Beds Category')['Child Mortality'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Hospital Beds Category', y='Child Mortality', data=grouped_df, palette='viridis', ax=ax)
    ax.set_title("Mortalité infantile moyenne par nombre de lits d'hôpitaux")
    ax.set_xlabel("Nombre de lits d'hôpitaux pour 1000 habitants")
    ax.set_ylabel("Mortalité infantile moyenne")
    st.pyplot(fig)

    st.subheader("Relation entre le nombre de lits d'hôpitaux et la mortalité infantile (moyenne par pays)")
    df_avg = df.groupby('Country', as_index=False).mean(numeric_only=True)
    fig = px.scatter(df_avg, x='Hospital Beds', y='Child Mortality', color='Country',
                    title="Relation entre le nombre de lits d'hôpitaux et la mortalité infantile",
                    labels={'Hospital Beds': "Lits d'hôpitaux (par 1000 habitants)",
                            'Child Mortality': "Mortalité infantile (pour 1000 naissances)"})
    st.plotly_chart(fig)

with tab4:
    st.subheader("Impact de la densité de population sur l'espérance de vie (Moyenne par tranche)")
    if {'Population Density', 'Life Expectancy', 'Child Mortality'}.issubset(df.columns):
        df_filtered = df[['Population Density', 'Life Expectancy', 'Child Mortality']].dropna()
        df_filtered["Density Group"] = pd.qcut(df_filtered["Population Density"], q=10, labels=False)
        density_avg = df_filtered.groupby("Density Group")[["Population Density", "Life Expectancy", "Child Mortality"]].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=density_avg, x="Population Density", y="Life Expectancy", marker="o", color="blue", ax=ax)
        ax.set_xlabel("Densité de population (habitants/km²)")
        ax.set_ylabel("Espérance de vie (années)")
        ax.set_title("Impact de la densité de population sur l'espérance de vie (Moyenne par tranche)")
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("Impact de la densité de population sur la mortalité infantile (Moyenne par tranche)")
    if {'Population Density', 'Life Expectancy', 'Child Mortality'}.issubset(df.columns):
        df_filtered = df[['Population Density', 'Life Expectancy', 'Child Mortality']].dropna()
        df_filtered["Density Group"] = pd.qcut(df_filtered["Population Density"], q=10, labels=False)
        density_avg = df_filtered.groupby("Density Group")[["Population Density", "Life Expectancy", "Child Mortality"]].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=density_avg, x="Population Density", y="Child Mortality", marker="o", color="red", ax=ax)
        ax.set_xlabel("Densité de population (habitants/km²)")
        ax.set_ylabel("Mortalité infantile (pour 1 000 naissances)")
        ax.set_title("Impact de la densité de population sur la mortalité infantile (Moyenne par tranche)")
        ax.grid(True)
        st.pyplot(fig)

   
    

    st.subheader("Corrélation entre la population, l'espérance de vie et les lits d'hôpitaux")
    df_filtered = df[['population', 'Life Expectancy', 'Hospital Beds', 'Health Expenditure']].dropna()
    correlation_matrix = df_filtered.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Corrélation entre la population, l'espérance de vie et les lits d'hôpitaux")
    st.pyplot(fig)

with tab5: 
    pays_comparaison = ["Malta", "China"]
    st.subheader("Évolution du PIB par habitant : Chine vs Malte")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="GDP per Capita", hue="Country", marker="o", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("PIB par habitant (USD)")
        ax.set_title("Évolution du PIB par habitant : Chine vs Malte")
        ax.grid(True)
        ax.legend(title="Pays")
        st.pyplot(fig)

    pays_comparaison = ["Malta", "China"]
    st.subheader("Comparaison des dépenses de santé : Chine vs Malte")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax1 = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="Health Expenditure", hue="Country", marker="o", ax=ax1)
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Dépenses de santé (% du PIB)", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.set_title("Comparaison des dépenses de santé : Chine vs Malte")
        ax1.grid(True)
        ax1.legend(title="Pays")
        st.pyplot(fig)

    pays_comparaison = ["Malta", "China"]
    st.subheader("Comparaison des lits d'hôpitaux : Chine vs Malte")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="Hospital Beds", hue="Country", marker="s", linestyle="dashed", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("Lits d'hôpitaux pour 1 000 habitants")
        ax.set_title("Comparaison des lits d'hôpitaux : Chine vs Malte")
        ax.grid(True)
        ax.legend(title="Pays")
        st.pyplot(fig)

    pays_comparaison = ["Malta", "China"]
    st.subheader("Évolution de l'espérance de vie : Malte vs Chine")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="Life Expectancy", hue="Country", marker="o", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("Espérance de vie (années)")
        ax.set_title("Évolution de l'espérance de vie : Malte vs Chine")
        ax.grid(True)
        ax.legend(title="Pays")
        st.pyplot(fig)

    pays_comparaison = ["France", "China", "United States"]
    st.subheader("Évolution du PIB par habitant : Chine vs France vs USA")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="GDP per Capita", hue="Country", marker="o", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("PIB par habitant (USD)")
        ax.set_title("Évolution du PIB par habitant : Chine vs France vs USA")
        ax.grid(True)
        ax.legend(title="Pays")
        st.pyplot(fig)

    pays_comparaison = ["China", "France", "United States"]
    st.subheader("Comparaison des dépenses de santé : Chine vs France vs USA")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax1 = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="Health Expenditure", hue="Country", marker="o", ax=ax1)
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Dépenses de santé (% du PIB)", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.set_title("Comparaison des dépenses de santé : Chine vs France vs USA")
        ax1.grid(True)
        ax1.legend(title="Pays")
        st.pyplot(fig)

    pays_comparaison = ["China", "France", "United States"]
    st.subheader("Comparaison des lits d'hôpitaux : Chine vs France vs USA")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="Hospital Beds", hue="Country", marker="s", linestyle="dashed", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("Lits d'hôpitaux pour 1 000 habitants")
        ax.set_title("Comparaison des lits d'hôpitaux : Chine vs France vs USA")
        ax.grid(True)
        ax.legend(title="Pays")
        st.pyplot(fig)

    pays_comparaison = ["China", "France", "United States"]
    st.subheader("Évolution de l'espérance de vie : Chine vs France vs USA")
    if {'Country', 'Year', 'GDP per Capita', 'Health Expenditure', 'Hospital Beds', 'Life Expectancy'}.issubset(df.columns):
        df_filtered = df[df["Country"].isin(pays_comparaison)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="Year", y="Life Expectancy", hue="Country", marker="o", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("Espérance de vie (années)")
        ax.set_title("Évolution de l'espérance de vie : Chine vs France vs USA")
        ax.grid(True)
        ax.legend(title="Pays")
        st.pyplot(fig)

    st.subheader("Corrélation entre les dépenses de santé, l'espérance de vie et les lits d'hôpitaux")
    df_filtered = df[['Life Expectancy', 'Hospital Beds', 'Health Expenditure']].dropna()
    correlation_matrix = df_filtered.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Corrélation entre les dépenses de santé, l'espérance de vie et les lits d'hôpitaux")
    st.pyplot(fig)