/**
 * AegisCredit.AI - Core Client Controller
 * Manages Chart.js visualizers, real-time ML inference, SQL querying, and stress testing.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State storage
    let currentAuditEntries = [];
    let currentSqlRows = [];
    let currentSqlCols = [];
    let samplePersonas = [];
    let gradeChartInstance = null;
    let exposureChartInstance = null;
    let rocChartInstance = null;
    let featChartInstance = null;

    // View Titles Configuration
    const VIEW_TITLES = {
        'tab-overview': {
            title: 'Portfolio Analytics & Risk Overview',
            desc: 'Real-time credit exposure metrics and empirical default probability analytics'
        },
        'tab-assessor': {
            title: 'Loan Risk Decision Support Console',
            desc: 'Predictive default probability scoring, rule-based tiering, and explainability'
        },
        'tab-sql': {
            title: 'SQL Credit Analytics Console',
            desc: 'Direct relational database querying and audit inspection in SQLite'
        },
        'tab-models': {
            title: 'Machine Learning Model Benchmarks',
            desc: 'Validation curves, ROC-AUC comparison, and feature importance rankings'
        },
        'tab-stress': {
            title: 'Macroeconomic Scenario Stress Testing',
            desc: 'Simulate severe market interest rate hikes, wage compression, and credit shocks'
        },
        'tab-pipeline': {
            title: 'Pipeline Architecture & Methodology',
            desc: 'Technical implementation details and mathematical formulation'
        }
    };

    // 1. Navigation Controller
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const viewTitleElem = document.getElementById('current-view-title');
    const viewDescElem = document.getElementById('current-view-desc');

    function switchTab(tabId) {
        navItems.forEach(item => {
            if (item.getAttribute('data-tab') === tabId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        tabPanes.forEach(pane => {
            if (pane.id === tabId) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });

        if (VIEW_TITLES[tabId]) {
            viewTitleElem.textContent = VIEW_TITLES[tabId].title;
            viewDescElem.textContent = VIEW_TITLES[tabId].desc;
        }

        // Trigger chart resizes if switching to tab-models or tab-overview
        if (tabId === 'tab-models' && !rocChartInstance) {
            loadModelPerformance();
        }
    }

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    document.getElementById('btn-quick-assess').addEventListener('click', () => {
        switchTab('tab-assessor');
    });

    // 2. Live UTC Clock
    function updateClock() {
        const now = new Date();
        const timeStr = now.toISOString().substring(11, 19) + ' UTC';
        document.getElementById('live-clock').textContent = timeStr;
    }
    setInterval(updateClock, 1000);
    updateClock();

    // 3. Load Portfolio Analytics (Tab 1)
    async function loadPortfolioStats() {
        try {
            const res = await fetch('/api/portfolio_stats');
            const data = await res.json();
            if (!data.success) return;

            const m = data.metrics;
            document.getElementById('kpi-exposure').textContent = `$${m.total_exposure_usd.toLocaleString()}`;
            document.getElementById('kpi-total-loans').textContent = m.total_loans.toLocaleString();
            document.getElementById('kpi-default-rate').textContent = `${m.overall_default_rate}%`;
            document.getElementById('kpi-avg-fico').textContent = m.avg_credit_score.toFixed(1);

            renderPortfolioCharts(data.grades_data);
            renderAuditLog(data.recent_evaluations);
        } catch (err) {
            console.error('Error fetching portfolio stats:', err);
        }
    }

    function renderPortfolioCharts(grades) {
        // Chart 1: Default Rate by Grade Bar Chart
        const ctxBar = document.getElementById('chart-grade-defaults').getContext('2d');
        if (gradeChartInstance) gradeChartInstance.destroy();

        gradeChartInstance = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: grades.labels.map(g => `Grade ${g}`),
                datasets: [{
                    label: 'Default Rate (%)',
                    data: grades.default_rates,
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.75)',
                        'rgba(59, 130, 246, 0.75)',
                        'rgba(245, 158, 11, 0.75)',
                        'rgba(249, 115, 22, 0.75)',
                        'rgba(239, 68, 68, 0.85)',
                        'rgba(225, 29, 72, 0.95)'
                    ],
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Default Probability: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 80,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', callback: v => `${v}%` }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });

        // Chart 2: Capital Allocation Doughnut
        const ctxDonut = document.getElementById('chart-exposure-donut').getContext('2d');
        if (exposureChartInstance) exposureChartInstance.destroy();

        exposureChartInstance = new Chart(ctxDonut, {
            type: 'doughnut',
            data: {
                labels: grades.labels.map(g => `Grade ${g}`),
                datasets: [{
                    data: grades.exposure,
                    backgroundColor: [
                        '#10b981', '#3b82f6', '#f59e0b', '#f97316', '#ef4444', '#be123c'
                    ],
                    borderWidth: 2,
                    borderColor: '#0b101d'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 12 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Exposure: $${(ctx.raw / 1000000).toFixed(2)}M`
                        }
                    }
                },
                cutout: '68%'
            }
        });
    }

    function renderAuditLog(entries) {
        currentAuditEntries = entries || [];
        const tbody = document.querySelector('#table-audit-log tbody');
        if (!entries || entries.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No audit transactions recorded yet.</td></tr>';
            return;
        }

        tbody.innerHTML = entries.map(row => {
            const probPct = (row.default_probability * 100).toFixed(1);
            let badgeClass = 'badge-approve';
            if (row.recommended_decision.includes('REVIEW')) badgeClass = 'badge-review';
            if (row.recommended_decision.includes('REJECT')) badgeClass = 'badge-reject';

            return `
                <tr>
                    <td><code>#${row.eval_id}</code></td>
                    <td><strong>${row.applicant_id}</strong></td>
                    <td>${row.model_version}</td>
                    <td><span class="${row.default_probability > 0.4 ? 'text-red' : 'text-green'}">${probPct}%</span></td>
                    <td>${row.risk_tier}</td>
                    <td><span class="badge ${badgeClass}">${row.recommended_decision}</span></td>
                    <td><small class="text-muted">${row.evaluated_at}</small></td>
                </tr>
            `;
        }).join('');
    }

    document.getElementById('btn-refresh-audit').addEventListener('click', loadPortfolioStats);

    // 4. Loan Assessor & Inference Engine (Tab 2)
    async function loadSamplePersonas() {
        try {
            const res = await fetch('/api/sample_applicants');
            const data = await res.json();
            if (data.success) {
                samplePersonas = data.samples;
            }
        } catch (err) {
            console.error('Failed to load sample personas:', err);
        }
    }

    function fillFormWithPersona(index) {
        if (!samplePersonas[index]) return;
        const p = samplePersonas[index];

        document.getElementById('inp-age').value = p.age;
        document.getElementById('inp-income').value = p.annual_income;
        document.getElementById('inp-emp-len').value = p.employment_length;
        document.getElementById('inp-home').value = p.home_ownership;
        document.getElementById('inp-intent').value = p.loan_intent;
        document.getElementById('inp-amount').value = p.loan_amount;
        document.getElementById('inp-term').value = p.loan_term_months;
        document.getElementById('inp-grade').value = p.loan_grade;
        document.getElementById('inp-rate').value = p.interest_rate;
        document.getElementById('inp-credit-score').value = p.credit_score;
        document.getElementById('inp-dti').value = p.dti;
        document.getElementById('inp-util').value = p.revolving_utilization;
        document.getElementById('inp-hist-len').value = p.credit_history_length;
        document.getElementById('inp-delinq').value = p.delinquencies_2yrs;
        document.getElementById('inp-pub-rec').value = p.public_records;
        document.getElementById('inp-inq').value = p.inquiries_last_6m;

        // Auto trigger assessment
        executeAssessment(p.applicant_id);
    }

    const presetChips = document.querySelectorAll('.preset-chip');
    presetChips.forEach(chip => {
        chip.addEventListener('click', () => {
            presetChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            const pIndex = parseInt(chip.getAttribute('data-preset'), 10);
            fillFormWithPersona(pIndex);
        });
    });

    async function executeAssessment(customAppId = null) {
        const payload = {
            applicant_id: customAppId || `LN-APP-${Math.floor(1000 + Math.random() * 9000)}`,
            age: parseInt(document.getElementById('inp-age').value, 10),
            annual_income: parseFloat(document.getElementById('inp-income').value),
            employment_length: parseInt(document.getElementById('inp-emp-len').value, 10),
            home_ownership: document.getElementById('inp-home').value,
            loan_intent: document.getElementById('inp-intent').value,
            loan_amount: parseFloat(document.getElementById('inp-amount').value),
            loan_term_months: parseInt(document.getElementById('inp-term').value, 10),
            loan_grade: document.getElementById('inp-grade').value,
            interest_rate: parseFloat(document.getElementById('inp-rate').value),
            credit_score: parseInt(document.getElementById('inp-credit-score').value, 10),
            dti: parseFloat(document.getElementById('inp-dti').value),
            revolving_utilization: parseFloat(document.getElementById('inp-util').value),
            credit_history_length: parseInt(document.getElementById('inp-hist-len').value, 10),
            delinquencies_2yrs: parseInt(document.getElementById('inp-delinq').value, 10),
            public_records: parseInt(document.getElementById('inp-pub-rec').value, 10),
            inquiries_last_6m: parseInt(document.getElementById('inp-inq').value, 10),
            open_accounts: 10
        };

        const btn = document.getElementById('btn-run-inference');
        btn.disabled = true;
        btn.innerHTML = 'Computing Risk Score...';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                renderAssessmentResult(data.data);
                // Also refresh audit table in background
                loadPortfolioStats();
            } else {
                alert('Prediction Error: ' + data.error);
            }
        } catch (err) {
            console.error('Inference request failed:', err);
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Execute Risk Assessment
            `;
        }
    }

    function renderAssessmentResult(res) {
        document.getElementById('res-prob-pct').textContent = `${res.default_probability_pct}%`;
        document.getElementById('res-repay-prob').textContent = `${res.repayment_probability_pct}%`;
        document.getElementById('res-risk-tier').textContent = res.risk_tier;
        document.getElementById('res-offered-rate').textContent = `${res.recommended_rate}%`;
        document.getElementById('res-max-amt').textContent = `$${res.max_approved_amount.toLocaleString()}`;
        document.getElementById('res-audit-id').textContent = `Audited ID: ${res.applicant_id}`;
        document.getElementById('res-model-badge').textContent = res.model_name;

        // Gauge update: circumference = 2 * PI * 50 ≈ 314
        const circumference = 314;
        const offset = circumference - (res.default_probability * circumference);
        const gaugeCircle = document.getElementById('gauge-circle');
        gaugeCircle.style.strokeDashoffset = offset;

        // Decision styling
        const badge = document.getElementById('res-decision-badge');
        const desc = document.getElementById('res-tier-desc');
        badge.className = 'decision-badge';

        if (res.decision_code === 'APPROVE') {
            badge.classList.add('badge-approve');
            badge.textContent = res.decision;
            gaugeCircle.style.stroke = '#10b981';
            desc.textContent = 'Application cleared prime underwriting requirements with low risk probability.';
        } else if (res.decision_code === 'REVIEW') {
            badge.classList.add('badge-review');
            badge.textContent = res.decision;
            gaugeCircle.style.stroke = '#f59e0b';
            desc.textContent = 'Credit profile exhibits moderate leverage or inquiries. Manual verification required.';
        } else {
            badge.classList.add('badge-reject');
            badge.textContent = res.decision;
            gaugeCircle.style.stroke = '#ef4444';
            desc.textContent = 'Default probability exceeds maximum risk tolerance limit. Adverse action generated.';
        }

        // Explainable risk factors
        const factorsContainer = document.getElementById('res-factors-list');
        if (res.risk_factors && res.risk_factors.length > 0) {
            factorsContainer.innerHTML = res.risk_factors.map(f => {
                const isPos = f.impact === 'positive';
                return `
                    <div class="factor-item ${isPos ? 'factor-pos' : 'factor-neg'}">
                        <span class="factor-icon">${isPos ? '✓' : '⚠'}</span>
                        <span class="factor-name">${f.factor}</span>
                        <span class="factor-impact">${isPos ? 'Reduces Risk' : 'Elevates Risk'}</span>
                    </div>
                `;
            }).join('');
        }
    }

    document.getElementById('btn-run-inference').addEventListener('click', () => executeAssessment());
    document.getElementById('btn-reset-form').addEventListener('click', () => {
        document.getElementById('loan-assessment-form').reset();
    });

    // 5. SQL Query Studio (Tab 3)
    const sqlTextarea = document.getElementById('sql-input-query');
    const sqlStatsBar = document.getElementById('sql-stats-bar');
    const btnExecuteSql = document.getElementById('btn-execute-sql');
    const btnExportCsv = document.getElementById('btn-export-csv');

    // Snippets
    document.querySelectorAll('.btn-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-sql');
            sqlTextarea.value = query;
            runSqlQuery();
        });
    });

    async function runSqlQuery() {
        const query = sqlTextarea.value.trim();
        if (!query) return;

        btnExecuteSql.disabled = true;
        sqlStatsBar.textContent = 'Executing query in SQLite database...';

        try {
            const res = await fetch('/api/sql_query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            const data = await res.json();

            if (data.success) {
                currentSqlCols = data.columns;
                currentSqlRows = data.rows;
                sqlStatsBar.innerHTML = `✓ Executed in <strong>${data.execution_time_ms} ms</strong> | Returned <strong>${data.row_count}</strong> rows.`;
                btnExportCsv.disabled = false;
                renderSqlTable(data.columns, data.rows);
            } else {
                sqlStatsBar.innerHTML = `<span class="text-red">Query Failed: ${data.error}</span>`;
                btnExportCsv.disabled = true;
            }
        } catch (err) {
            sqlStatsBar.innerHTML = `<span class="text-red">Execution Error: ${err.message}</span>`;
        } finally {
            btnExecuteSql.disabled = false;
        }
    }

    function renderSqlTable(columns, rows) {
        const table = document.getElementById('table-sql-results');
        if (!columns || columns.length === 0) {
            table.innerHTML = '<thead><tr><th>Empty Result</th></tr></thead><tbody><tr><td>No data returned.</td></tr></tbody>';
            return;
        }

        const thead = `<thead><tr>${columns.map(c => `<th>${c}</th>`).join('')}</tr></thead>`;
        const tbody = `<tbody>${rows.map(r => `
            <tr>${columns.map(c => `<td>${r[c] !== null ? r[c] : 'NULL'}</td>`).join('')}</tr>
        `).join('')}</tbody>`;

        table.innerHTML = thead + tbody;
    }

    btnExecuteSql.addEventListener('click', runSqlQuery);

    btnExportCsv.addEventListener('click', () => {
        if (!currentSqlRows || currentSqlRows.length === 0) return;
        const csvContent = [
            currentSqlCols.join(','),
            ...currentSqlRows.map(row => currentSqlCols.map(c => `"${row[c] !== null ? row[c] : ''}"`).join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sql_export_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // 6. Model Performance (Tab 4)
    async function loadModelPerformance() {
        try {
            const res = await fetch('/api/model_metrics');
            const data = await res.json();
            if (!data.success) return;

            const models = data.models;
            // Fill card metrics
            if (models['Logistic Regression']) {
                const lr = models['Logistic Regression'];
                document.getElementById('m-lr-auc').textContent = lr.roc_auc;
                document.getElementById('m-lr-pr').textContent = lr.pr_auc;
                document.getElementById('m-lr-rec').textContent = `${(lr.recall * 100).toFixed(1)}%`;
            }
            if (models['Random Forest']) {
                const rf = models['Random Forest'];
                document.getElementById('m-rf-auc').textContent = rf.roc_auc;
                document.getElementById('m-rf-pr').textContent = rf.pr_auc;
                document.getElementById('m-rf-rec').textContent = `${(rf.recall * 100).toFixed(1)}%`;
            }
            if (models['LightGBM']) {
                const lgb = models['LightGBM'];
                document.getElementById('m-lgb-auc').textContent = lgb.roc_auc;
                document.getElementById('m-lgb-pr').textContent = lgb.pr_auc;
                document.getElementById('m-lgb-rec').textContent = `${(lgb.recall * 100).toFixed(1)}%`;
            }

            renderRocChart(models);
            renderFeatureImportance(data.metadata.top_features);
        } catch (err) {
            console.error('Failed to load model metrics:', err);
        }
    }

    function renderRocChart(models) {
        const ctxRoc = document.getElementById('chart-roc-curves').getContext('2d');
        if (rocChartInstance) rocChartInstance.destroy();

        const datasets = [];
        const colors = {
            'Logistic Regression': '#10b981',
            'Random Forest': '#3b82f6',
            'LightGBM': '#f59e0b'
        };

        for (const [name, m] of Object.entries(models)) {
            if (m.roc_points) {
                datasets.push({
                    label: `${name} (AUC = ${m.roc_auc})`,
                    data: m.roc_points.map(p => ({ x: p.fpr, y: p.tpr })),
                    borderColor: colors[name] || '#6366f1',
                    borderWidth: 2.5,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0
                });
            }
        }

        // Random guess diagonal
        datasets.push({
            label: 'Random Baseline (AUC = 0.50)',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        });

        rocChartInstance = new Chart(ctxRoc, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' },
                        min: 0, max: 1
                    },
                    y: {
                        type: 'linear',
                        title: { display: true, text: 'True Positive Rate (Sensitivity / Recall)', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' },
                        min: 0, max: 1
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } }
                    }
                }
            }
        });
    }

    function renderFeatureImportance(topFeatures) {
        const ctxFeat = document.getElementById('chart-feature-importance').getContext('2d');
        if (featChartInstance) featChartInstance.destroy();

        const labels = Object.keys(topFeatures || {}).reverse();
        const values = Object.values(topFeatures || {}).reverse();

        featChartInstance = new Chart(ctxFeat, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Relative Weight (%)',
                    data: values,
                    backgroundColor: 'rgba(56, 189, 248, 0.75)',
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', callback: v => `${v}%` }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#cbd5e1', font: { family: 'JetBrains Mono', size: 11 } }
                    }
                }
            }
        });
    }

    // 7. Stress Testing (Tab 5)
    const stressRateSlider = document.getElementById('stress-rate');
    const stressIncomeSlider = document.getElementById('stress-income');
    const stressDtiSlider = document.getElementById('stress-dti');

    stressRateSlider.addEventListener('input', e => {
        document.getElementById('stress-rate-val').textContent = `+${e.target.value}%`;
    });
    stressIncomeSlider.addEventListener('input', e => {
        document.getElementById('stress-income-val').textContent = `${e.target.value}%`;
    });
    stressDtiSlider.addEventListener('input', e => {
        document.getElementById('stress-dti-val').textContent = `+${e.target.value}%`;
    });

    document.getElementById('btn-run-stress').addEventListener('click', async () => {
        const payload = {
            interest_rate_delta: parseFloat(stressRateSlider.value),
            income_shock_pct: parseFloat(stressIncomeSlider.value),
            dti_shock_pct: parseFloat(stressDtiSlider.value)
        };

        const btn = document.getElementById('btn-run-stress');
        btn.disabled = true;
        btn.innerHTML = 'Simulating Portfolio Stress Dynamics...';

        try {
            const res = await fetch('/api/stress_test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                document.getElementById('res-base-def').textContent = `${data.base_default_rate}%`;
                document.getElementById('res-stress-def').textContent = `${data.stressed_default_rate}%`;
                document.getElementById('res-delta-def').textContent = `+${data.delta_rate}%`;
                document.getElementById('res-loss-surge').textContent = `+$${data.loss_increase_usd.toLocaleString()}`;
            }
        } catch (err) {
            console.error('Stress test failed:', err);
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg>
                Execute Stress Test Simulation
            `;
        }
    });

    // Initial Boots
    loadPortfolioStats();
    loadSamplePersonas().then(() => {
        // Pre-run first persona as initial showcase
        if (samplePersonas.length > 0) {
            fillFormWithPersona(0);
        }
    });
});
