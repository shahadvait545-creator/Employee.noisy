"""
Employee Skill Intelligence Dashboard
Run with: streamlit run dashboard.py
Install:  pip install streamlit pandas seaborn matplotlib openai plotly
"""

import os, re, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Skill Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border: 1px solid #3a3a5c;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    color: white;
}
.metric-card .label { font-size: 12px; color: #a0a0c0; letter-spacing: 1px; text-transform: uppercase; }
.metric-card .value { font-size: 36px; font-weight: 700; color: #7c6bff; }
.metric-card .sub   { font-size: 12px; color: #888; margin-top: 4px; }

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    color: #1e1e2e; margin: 28px 0 12px;
    border-left: 4px solid #7c6bff;
    padding-left: 12px;
}
.badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin: 2px;
}
.badge-green  { background: #d4edda; color: #155724; }
.badge-red    { background: #f8d7da; color: #721c24; }
.badge-blue   { background: #cce5ff; color: #004085; }
.badge-orange { background: #fff3cd; color: #856404; }

.quality-bar-wrap { background: #eee; border-radius: 8px; height: 12px; margin: 8px 0; }
.quality-bar      { border-radius: 8px; height: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EMAIL_RE           = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
PHONE_10_DIGITS_RE = r'^\d{10}$'
EMPID_RE           = r'^E\d{3}$'

VALID_DESIGNATIONS = {
    'AI Intern', 'Data Analyst', 'Data Scientist', 'ML Engineer', 'Software Engineer'
}

CANON_SKILLS_MAP = {
    'ai': 'AI', 'ml': 'ML', 'python': 'Python', 'java': 'Java', 'sql': 'SQL',
    'cloud': 'Cloud', 'nlp': 'NLP', 'genai': 'GenAI', 'mlops': 'MLOps'
}
VALID_SKILLS = set(CANON_SKILLS_MAP.values())

ROLE_SKILL_MATRIX = {
    'Data Scientist'   : ['Python', 'ML', 'SQL', 'GenAI', 'NLP', 'Cloud'],
    'ML Engineer'      : ['Python', 'ML', 'Cloud', 'SQL', 'MLOps', 'GenAI'],
    'Software Engineer': ['Python', 'Java', 'SQL', 'Cloud', 'GenAI'],
    'Data Analyst'     : ['SQL', 'Python', 'Cloud', 'GenAI'],
    'AI Intern'        : ['Python', 'ML', 'GenAI'],
}

SKILL_RESOURCES = {
    'Python' : [("Python.org Docs",     "https://docs.python.org/3/tutorial/"),
                ("Kaggle Python",        "https://www.kaggle.com/learn/python"),
                ("Real Python",          "https://realpython.com")],
    'ML'     : [("fast.ai",             "https://www.fast.ai"),
                ("Google ML Crash",      "https://developers.google.com/machine-learning/crash-course"),
                ("Coursera ML (Andrew)", "https://www.coursera.org/learn/machine-learning")],
    'SQL'    : [("Mode SQL Tutorial",   "https://mode.com/sql-tutorial/"),
                ("SQLZoo",              "https://sqlzoo.net"),
                ("LeetCode SQL",        "https://leetcode.com/problemset/database/")],
    'Cloud'  : [("AWS Training",        "https://aws.amazon.com/training/"),
                ("Google Cloud Skills", "https://cloudskillsboost.google"),
                ("Azure Learn",         "https://learn.microsoft.com/en-us/azure/")],
    'GenAI'  : [("DeepLearning.AI",     "https://www.deeplearning.ai"),
                ("Google GenAI Path",   "https://cloudskillsboost.google/paths/118"),
                ("Hugging Face Course", "https://huggingface.co/learn")],
    'NLP'    : [("HF NLP Course",       "https://huggingface.co/learn/nlp-course"),
                ("Stanford NLP",        "https://web.stanford.edu/class/cs224n/"),
                ("NLTK Book",           "https://www.nltk.org/book/")],
    'MLOps'  : [("MLOps Zoomcamp",      "https://github.com/DataTalksClub/mlops-zoomcamp"),
                ("Made With ML",        "https://madewithml.com"),
                ("Weights & Biases",    "https://docs.wandb.ai")],
    'Java'   : [("Oracle Java Tuts",    "https://docs.oracle.com/javase/tutorial/"),
                ("Codecademy Java",     "https://www.codecademy.com/learn/learn-java"),
                ("Baeldung",            "https://www.baeldung.com")],
    'AI'     : [("Elements of AI",      "https://www.elementsofai.com"),
                ("AI For Everyone",     "https://www.coursera.org/learn/ai-for-everyone"),
                ("OpenAI Cookbook",     "https://cookbook.openai.com")],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_to_digits(s):
    if pd.isna(s): return np.nan
    digits = re.sub(r'\D', '', str(s))
    return digits if digits else np.nan

def clean_and_split_skills(cell):
    if pd.isna(cell): return []
    s = str(cell).replace(' - ', ',').replace('-', ',')
    parts = [p.strip() for p in s.split(',') if p.strip()]
    out, seen = [], set()
    for p in parts:
        val = CANON_SKILLS_MAP.get(p.lower(), p)
        if val not in seen:
            seen.add(val)
            out.append(val)
    return out

def has_invalid_skill_tokens(cell):
    if pd.isna(cell): return False
    s = str(cell).replace(' - ', ',').replace('-', ',')
    for p in [x.strip() for x in s.split(',') if x.strip()]:
        if CANON_SKILLS_MAP.get(p.lower(), p) not in VALID_SKILLS:
            return True
    return False

@st.cache_data
def process_dataframe(raw: pd.DataFrame):
    work = raw.copy()
    work.columns = work.columns.str.strip()

    work['invalid_email']       = ~work['Email_ID'].astype(str).str.match(EMAIL_RE, na=True)
    work['__phone_digits']      = work['Phone Number'].apply(strip_to_digits)
    work['invalid_phone']       = ~work['__phone_digits'].astype(str).str.match(PHONE_10_DIGITS_RE, na=False)
    work['invalid_employee_id'] = ~work['Employee_ID'].astype(str).str.match(EMPID_RE, na=False)
    NAME_RE = r"^[A-Za-z .'\-]{3,}$"
    work['invalid_name']        = ~work['Name'].astype(str).str.match(NAME_RE, na=False)
    work['invalid_designation'] = ~work['Designation'].isin(VALID_DESIGNATIONS)
    work['has_missing']         = work.isna().any(axis=1) | (work.astype(str).eq('').any(axis=1))

    skills_list              = work['Skills'].apply(clean_and_split_skills)
    work['Skills_canonical'] = skills_list.apply(lambda L: ', '.join(L))
    work['Skills_list']      = skills_list
    work['skill_count']      = skills_list.apply(len)

    work['invalid_skills_raw']    = work['Skills'].apply(has_invalid_skill_tokens)
    work['no_skills_after_parse'] = skills_list.apply(lambda L: len(L) == 0)

    flag_cols = [
        'invalid_email', 'invalid_phone', 'invalid_employee_id', 'invalid_name',
        'invalid_designation', 'invalid_skills_raw', 'no_skills_after_parse', 'has_missing'
    ]
    work['flag_count'] = work[flag_cols].sum(axis=1)
    total_checks = len(work) * len(flag_cols)
    total_issues = work[flag_cols].sum().sum()
    work['data_quality_score'] = round((1 - total_issues / total_checks) * 100, 1)

    return work, flag_cols, skills_list

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Skill Intelligence")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])
    st.markdown("---")
    _secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    openai_key = st.text_input(
        "Groq API Key (for AI Report)",
        value=_secret_key,
        type="password",
        placeholder="gsk_...  (free at console.groq.com)",
    )
    st.markdown("---")
    st.markdown("**Navigation**")
    section = st.radio("Go to", [
        "📊 Overview",
        "🔍 Data Quality",
        "🎯 Skills Analysis",
        "👤 Employee Explorer",
        "🏆 Top Candidates",
        "🤖 AI Skill Gap Report",
        "📚 Study Resources",
    ])

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Employee Skill Intelligence Dashboard")
st.markdown("Upload your employee CSV to get started.")

if uploaded is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

raw_df = pd.read_csv(uploaded)
work, flag_cols, skills_list = process_dataframe(raw_df)
skills_exploded = skills_list.explode()

role_sizes = work['Designation'].value_counts().to_dict()
role_skill_counts = {}
for role in role_sizes:
    sub  = work[work['Designation'] == role]
    expl = sub['Skills_canonical'].str.split(', ').explode().dropna()
    role_skill_counts[role] = expl.value_counts().to_dict()

role_skill_coverage = {}
for role, req_skills in ROLE_SKILL_MATRIX.items():
    size = role_sizes.get(role, 0)
    cov  = {}
    for sk in req_skills:
        have    = role_skill_counts.get(role, {}).get(sk, 0)
        cov[sk] = round(have / size * 100, 1) if size else 0
    role_skill_coverage[role] = cov

dqs = work['data_quality_score'].iloc[0]

# ═══════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════
if section == "📊 Overview":
    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Total Employees", len(work),                             "records in dataset"),
        ("Unique Roles",    work['Designation'].nunique(),         "active designations"),
        ("Unique Skills",   skills_exploded.nunique(),             "distinct skills found"),
        ("Data Quality",    f"{dqs}%",                            "of records are clean"),
        ("Flagged Records", int(work['flag_count'].gt(0).sum()),   "have ≥1 issue"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4,c5], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            names=work['Designation'].value_counts().index,
            values=work['Designation'].value_counts().values,
            title="Designation Distribution",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hole=0.45,
        )
        fig.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=work['Education'].value_counts().index,
            y=work['Education'].value_counts().values,
            title="Education Distribution",
            labels={'x': 'Education', 'y': 'Count'},
            color_discrete_sequence=['#7c6bff'],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Role Summary</div>', unsafe_allow_html=True)
    for role, req_skills in ROLE_SKILL_MATRIX.items():
        size = role_sizes.get(role, 0)
        if size == 0:
            continue
        with st.expander(f"**{role}** — {size} employees"):
            cov = role_skill_coverage.get(role, {})
            cols = st.columns(len(req_skills))
            for i, sk in enumerate(req_skills):
                pct = cov.get(sk, 0)
                color = "#2ecc71" if pct >= 75 else "#f39c12" if pct >= 40 else "#e74c3c"
                cols[i].metric(sk, f"{pct}%")

# ═══════════════════════════════════════════════════════════
# 2. DATA QUALITY
# ═══════════════════════════════════════════════════════════
elif section == "🔍 Data Quality":
    st.markdown('<div class="section-header">Data Quality Score</div>', unsafe_allow_html=True)

    dqs_color = "#2ecc71" if dqs >= 80 else "#f39c12" if dqs >= 60 else "#e74c3c"
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dqs,
        title={'text': "Overall Data Quality Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar' : {'color': dqs_color},
            'steps': [
                {'range': [0,  60], 'color': '#fde8e8'},
                {'range': [60, 80], 'color': '#fff3cd'},
                {'range': [80, 100],'color': '#d4edda'},
            ],
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown('<div class="section-header">Flag Breakdown</div>', unsafe_allow_html=True)
    flag_df = work[flag_cols].sum().reset_index()
    flag_df.columns = ['Flag', 'Count']
    flag_df = flag_df.sort_values('Count', ascending=False)

    fig = px.bar(flag_df, x='Flag', y='Count',
                 color='Count', color_continuous_scale='Reds',
                 title="Issues per Flag Type")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Flagged Rows</div>', unsafe_allow_html=True)
    flagged = work[work['flag_count'] > 0].copy()
    show_cols = [c for c in ['Employee_ID','Name','Designation','flag_count'] + flag_cols if c in work.columns]
    st.dataframe(flagged[show_cols].sort_values('flag_count', ascending=False), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# 3. SKILLS ANALYSIS
# ═══════════════════════════════════════════════════════════
elif section == "🎯 Skills Analysis":
    st.markdown('<div class="section-header">Skill Frequency Count</div>', unsafe_allow_html=True)

    freq = skills_exploded.value_counts().reset_index()
    freq.columns = ['Skill', 'Count']
    fig = px.bar(freq, x='Skill', y='Count',
                 color='Count', color_continuous_scale='Viridis',
                 title="Skill Frequency Across All Employees")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Skills by Designation</div>', unsafe_allow_html=True)
    desg_skill = []
    for role in work['Designation'].unique():
        sub  = work[work['Designation'] == role]
        expl = sub['Skills_canonical'].str.split(', ').explode().dropna()
        for sk, cnt in expl.value_counts().items():
            desg_skill.append({'Role': role, 'Skill': sk, 'Count': cnt})

    ds_df = pd.DataFrame(desg_skill)
    if not ds_df.empty:
        fig = px.bar(ds_df, x='Skill', y='Count', color='Role',
                     barmode='group', title="Skills by Role",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Skill Coverage Heatmap</div>', unsafe_allow_html=True)
    roles_present = [r for r in ROLE_SKILL_MATRIX if r in role_sizes]
    all_skills    = sorted(VALID_SKILLS)
    heat_data = []
    for role in roles_present:
        row = []
        for sk in all_skills:
            pct = role_skill_coverage.get(role, {}).get(sk, 0)
            row.append(pct)
        heat_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=heat_data, x=all_skills, y=roles_present,
        colorscale='RdYlGn', zmin=0, zmax=100,
        text=[[f"{v}%" for v in row] for row in heat_data],
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Skill Coverage % by Role (heatmap)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Skills by Strength</div>', unsafe_allow_html=True)
    strength = freq.copy()
    strength['Strength'] = pd.cut(strength['Count'],
                                   bins=[0, 3, 7, 100],
                                   labels=['Low', 'Medium', 'High'])
    for level in ['High', 'Medium', 'Low']:
        grp = strength[strength['Strength'] == level]
        if not grp.empty:
            color = {'High': 'badge-green', 'Medium': 'badge-orange', 'Low': 'badge-red'}[level]
            badges = " ".join(f'<span class="badge {color}">{s}</span>' for s in grp['Skill'])
            st.markdown(f"**{level} Strength:** {badges}", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 4. EMPLOYEE EXPLORER
# ═══════════════════════════════════════════════════════════
elif section == "👤 Employee Explorer":
    st.markdown('<div class="section-header">Filter & Search Employees</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_role = st.multiselect("Designation", options=sorted(work['Designation'].dropna().unique()))
    with col2:
        sel_edu  = st.multiselect("Education", options=sorted(work['Education'].dropna().unique()))
    with col3:
        sel_skill = st.multiselect("Must have skill", options=sorted(VALID_SKILLS))

    name_search = st.text_input("Search by Name or Employee ID")

    filtered = work.copy()
    if sel_role:  filtered = filtered[filtered['Designation'].isin(sel_role)]
    if sel_edu:   filtered = filtered[filtered['Education'].isin(sel_edu)]
    if sel_skill:
        for sk in sel_skill:
            filtered = filtered[filtered['Skills_canonical'].str.contains(sk, na=False)]
    if name_search:
        mask = (filtered['Name'].astype(str).str.contains(name_search, case=False, na=False) |
                filtered['Employee_ID'].astype(str).str.contains(name_search, case=False, na=False))
        filtered = filtered[mask]

    st.markdown(f"**{len(filtered)} employees found**")

    show_cols = [c for c in ['Employee_ID','Name','Email_ID','Designation','Education',
                              'Skills_canonical','skill_count','flag_count'] if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True)

    csv_out = filtered[show_cols].to_csv(index=False).encode()
    st.download_button("⬇️ Export Filtered CSV", csv_out, "filtered_employees.csv", "text/csv")

# ═══════════════════════════════════════════════════════════
# 5. TOP CANDIDATES
# ═══════════════════════════════════════════════════════════
elif section == "🏆 Top Candidates":
    st.markdown('<div class="section-header">Top Candidates by Skills</div>', unsafe_allow_html=True)

    target_role = st.selectbox("Select Role to Match", list(ROLE_SKILL_MATRIX.keys()))
    req = ROLE_SKILL_MATRIX[target_role]

    def match_score(row):
        skills = [s.strip() for s in str(row['Skills_canonical']).split(',')]
        return sum(1 for s in req if s in skills)

    candidates = work[work['Designation'] == target_role].copy()
    if candidates.empty:
        candidates = work.copy()

    candidates['match_score'] = candidates.apply(match_score, axis=1)
    candidates['match_pct']   = (candidates['match_score'] / len(req) * 100).round(1)
    top = candidates.sort_values(['match_score', 'flag_count'], ascending=[False, True]).head(20)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Required skills for {target_role}:** " +
                    " ".join(f'<span class="badge badge-blue">{s}</span>' for s in req),
                    unsafe_allow_html=True)

    show_cols = [c for c in ['Employee_ID','Name','Designation','Skills_canonical',
                              'match_score','match_pct','flag_count'] if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True)

    fig = px.bar(top.head(10), x='Name', y='match_pct',
                 color='match_pct', color_continuous_scale='Greens',
                 title=f"Top 10 Candidates — {target_role} (% skill match)")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# 6. AI SKILL GAP REPORT
# ═══════════════════════════════════════════════════════════
elif section == "🤖 AI Skill Gap Report":
    st.markdown('<div class="section-header">AI-Generated Skill Gap Report</div>', unsafe_allow_html=True)
    st.info("This uses Groq (free tier) to generate an intelligent analysis. Get your free API key at [console.groq.com](https://console.groq.com) and paste it in the sidebar.")

    if st.button("🚀 Generate Report"):
        key = openai_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            st.error("Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
        else:
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=key,
                    base_url="https://api.groq.com/openai/v1",
                )

                payload = {
                    "role_sizes"          : role_sizes,
                    "role_skill_counts"   : role_skill_counts,
                    "role_skill_coverage" : role_skill_coverage,
                    "top_skills"          : skills_exploded.value_counts().to_dict(),
                }
                prompt = f"""
You are an HR analytics expert. Analyze this employee dataset:
1) Summarize the top 5 most common skills.
2) Identify role-wise skill gaps (coverage below 50%).
3) Recommend 2-3 specific online courses per gap skill.
4) Write a strategic workforce summary (2-3 sentences).
Use markdown, be concise and actionable.
JSON:
{json.dumps(payload, indent=2)}
"""
                with st.spinner("Generating AI report..."):
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                    )
                    report = resp.choices[0].message.content
                st.markdown(report)

            except Exception as e:
                st.error(f"Groq API error: {e}")

# ═══════════════════════════════════════════════════════════
# 7. STUDY RESOURCES
# ═══════════════════════════════════════════════════════════
elif section == "📚 Study Resources":
    st.markdown('<div class="section-header">Study Material by Skill</div>', unsafe_allow_html=True)
    st.markdown("Curated learning resources for every skill in the dataset.")

    for skill, links in SKILL_RESOURCES.items():
        with st.expander(f"📘 {skill}"):
            for name, url in links:
                st.markdown(f"- [{name}]({url})")

    st.markdown('<div class="section-header">Skill Gap Resources by Role</div>', unsafe_allow_html=True)
    for role, req_skills in ROLE_SKILL_MATRIX.items():
        size = role_sizes.get(role, 0)
        if size == 0:
            continue
        with st.expander(f"🎯 {role} — Gap Resources"):
            for sk in req_skills:
                pct = role_skill_coverage.get(role, {}).get(sk, 0)
                if pct < 60:
                    st.markdown(f"**{sk}** — ⚠️ {pct}% coverage — needs attention")
                    for name, url in SKILL_RESOURCES.get(sk, []):
                        st.markdown(f"  - [{name}]({url})")
                else:
                    st.markdown(f"**{sk}** — ✅ {pct}% coverage")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:12px;'>Employee Skill Intelligence Dashboard</div>",
    unsafe_allow_html=True
)
