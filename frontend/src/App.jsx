import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:5001";

const initialValues = {
  current_occupancy_beds: 200,
  previous_occupancy_beds: 195,
  two_weeks_ago_occupancy_beds: 190,
  total_beds_7_day_avg: 250,
  current_occupancy_rate: 0.8,
};

// Actual experiment results recorded for this project
const modelResults = [
  { model: "Persistence", MAE: 5.2151, RMSE: 11.1097, R2: 0.9961 },
  { model: "Linear Regression", MAE: 5.1802, RMSE: 10.94, R2: 0.9963 },
  { model: "Random Forest", MAE: 5.0579, RMSE: 11.0024, R2: 0.9962 },
];

const navItems = [
  { id: "overview", icon: "▦", label: "Overview" },
  { id: "forecast", icon: "↗", label: "Occupancy Forecast" },
  { id: "insights", icon: "▤", label: "Model Insights" },
  { id: "monitoring", icon: "◉", label: "Monitoring" },
];

function App() {
  const [page, setPage] = useState("overview");
  const [values, setValues] = useState(initialValues);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState("");
  const [metricsText, setMetricsText] = useState("");
  const [metricsError, setMetricsError] = useState("");

  const occupancy = Number(values.current_occupancy_beds);
  const totalBeds = Number(values.total_beds_7_day_avg);
  const occupancyRate = totalBeds > 0 ? (occupancy / totalBeds) * 100 : 0;

  const pageTitles = {
    overview: ["HOSPITAL OPERATIONS / OVERVIEW", "Capacity Dashboard"],
    forecast: ["PREDICTIVE ANALYTICS / FORECAST", "Occupancy Forecast"],
    insights: ["MODEL PERFORMANCE / ANALYSIS", "Model Insights"],
    monitoring: ["SYSTEM STATUS / OBSERVABILITY", "API Monitoring"],
  };

  const updateValue = (event) => {
    setValues({ ...values, [event.target.name]: event.target.value });
  };

  const handlePredict = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const payload = Object.fromEntries(
        Object.entries(values).map(([key, value]) => [key, Number(value)])
      );

      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed.");

      setPrediction(data);
      setHistory((previous) => [
        {
          id: Date.now(),
          date: new Date().toLocaleString(),
          current: payload.current_occupancy_beds,
          predicted: Number(data.predicted_occupancy_beds_next_week),
        },
        ...previous,
      ]);
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Cannot connect to the API. Check that the backend is running on port 5001."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const checkBackend = async () => {
    setHealthLoading(true);
    setHealthError("");
    setMetricsError("");

    try {
      const response = await fetch(`${API}/health`);
      const data = await response.json();
      if (!response.ok) throw new Error("Health endpoint returned an error.");
      setHealth(data);
    } catch (err) {
      setHealth(null);
      setHealthError(err.message);
    }

    try {
      const response = await fetch(`${API}/metrics`);
      if (!response.ok) throw new Error("Could not load Prometheus metrics.");
      const data = await response.text();
      setMetricsText(data);
    } catch (err) {
      setMetricsText("");
      setMetricsError(err.message);
    }

    setHealthLoading(false);
  };

  useEffect(() => {
    if (page === "monitoring") checkBackend();
  }, [page]);

  const metricLines = metricsText
    .split("\n")
    .filter((line) => line && !line.startsWith("#") && !line.startsWith(" "))
    .filter((line) =>
      /flask_http_request_total|flask_http_request_duration_seconds_count|process_resident_memory_bytes|python_info/.test(line)
    )
    .slice(0, 8);

  const forecastChartData = [...history]
    .slice(0, 8)
    .reverse()
    .map((item, index) => ({
      name: `Run ${index + 1}`,
      Actual: item.current,
      Predicted: item.predicted,
    }));

  const renderForecastForm = () => (
    <section className="content-grid">
      <article className="panel prediction-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">MACHINE LEARNING</p>
            <h2>Forecast Occupancy</h2>
            <p className="panel-subtitle">
              Enter recent occupancy information to predict next week's occupied beds.
            </p>
          </div>
          <span className="model-tag">RANDOM FOREST</span>
        </div>

        <form onSubmit={handlePredict}>
          <div className="form-grid">
            {[
              ["current_occupancy_beds", "Current occupied beds", 0, 1],
              ["previous_occupancy_beds", "Previous week's occupancy", 0, 1],
              ["two_weeks_ago_occupancy_beds", "Occupancy two weeks ago", 0, 1],
              ["total_beds_7_day_avg", "Total beds (7-day average)", 1, 1],
              ["current_occupancy_rate", "Current occupancy rate (0 to 1)", 0, 0.01],
            ].map(([name, label, min, step], index) => (
              <label key={name} className={index === 4 ? "full-width" : ""}>
                {label}
                <input
                  type="number"
                  name={name}
                  min={min}
                  max={name === "current_occupancy_rate" ? 1 : undefined}
                  step={step}
                  value={values[name]}
                  onChange={updateValue}
                  required
                />
                {index === 4 && (
                  <span className="input-help">
                    Example: 0.80 represents an occupancy rate of 80%.
                  </span>
                )}
              </label>
            ))}
          </div>

          {error && <div className="message error-message">{error}</div>}

          {prediction && (
            <div className="message success-message">
              <strong>Prediction generated successfully</strong>
              <p>
                Forecast:{" "}
                {Number(prediction.predicted_occupancy_beds_next_week).toFixed(2)} beds
                for the next 7 days.
              </p>
              <small>{prediction.warning}</small>
            </div>
          )}

          <button className="predict-button" type="submit" disabled={loading}>
            {loading ? "Generating forecast..." : "↗  Generate Prediction"}
          </button>
        </form>
      </article>

      <article className="panel insight-panel">
        <p className="eyebrow">CAPACITY INSIGHTS</p>
        <h2>Occupancy Snapshot</h2>
        <p className="panel-subtitle">A live view of the values entered for this forecast.</p>

        <div className="snapshot">
          <div className="snapshot-header">
            <span>Bed occupancy</span>
            <strong>{occupancyRate.toFixed(1)}%</strong>
          </div>
          <div className="large-progress">
            <div className="progress-fill" style={{ width: `${Math.min(occupancyRate, 100)}%` }} />
          </div>
          <p className="snapshot-caption">
            {occupancy.toLocaleString()} occupied out of {totalBeds.toLocaleString()} reference beds
          </p>
        </div>

        <div className="insight-row">
          <span className="insight-icon">01</span>
          <div>
            <strong>Historical trend inputs</strong>
            <p>{values.previous_occupancy_beds} beds last week · {values.two_weeks_ago_occupancy_beds} beds two weeks ago</p>
          </div>
        </div>
        <div className="insight-row">
          <span className="insight-icon">02</span>
          <div>
            <strong>Prediction horizon</strong>
            <p>Next 7 days · Model-based estimate</p>
          </div>
        </div>
        <div className="notice">
          <span>ⓘ</span>
          <p>Experimental forecasting tool. Predictions must not replace clinical or operational judgment.</p>
        </div>
      </article>
    </section>
  );

  const renderOverview = () => (
    <>
      <section className="welcome">
        <div>
          <p className="eyebrow light">PREDICTIVE HEALTHCARE ANALYTICS</p>
          <h2>Plan ahead. Care better.</h2>
          <p>Understand occupancy trends and forecast next week's bed requirements using machine learning.</p>
          <button className="welcome-button" onClick={() => setPage("forecast")}>
            Open occupancy forecast →
          </button>
        </div>
        <div className="welcome-symbol">✚</div>
      </section>

      <section className="stats-grid">
        <article className="stat-card">
          <div className="stat-top"><span>Currently Occupied</span><span className="stat-icon blue">▥</span></div>
          <h2>{occupancy.toLocaleString()}</h2>
          <p>Occupied beds · Input value</p>
        </article>
        <article className="stat-card">
          <div className="stat-top"><span>Total Bed Capacity</span><span className="stat-icon purple">▦</span></div>
          <h2>{totalBeds.toLocaleString()}</h2>
          <p>Available capacity reference</p>
        </article>
        <article className="stat-card">
          <div className="stat-top"><span>Current Occupancy Rate</span><span className="stat-icon orange">◷</span></div>
          <h2>{occupancyRate.toFixed(1)}%</h2>
          <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.min(occupancyRate, 100)}%` }} /></div>
        </article>
        <article className="stat-card prediction-stat">
          <div className="stat-top"><span>Next Week Forecast</span><span className="stat-icon green">↗</span></div>
          <h2>{prediction ? Number(prediction.predicted_occupancy_beds_next_week).toFixed(1) : "—"}</h2>
          <p>{prediction ? "Predicted occupied beds" : "Run a prediction to view"}</p>
        </article>
      </section>

      {renderForecastForm()}

      {history.length > 0 && (
        <section className="panel history-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">RECENT ACTIVITY</p><h2>Prediction History</h2></div>
            <button className="secondary-button" onClick={() => setHistory([])}>Clear history</button>
          </div>
          <HistoryTable history={history} />
        </section>
      )}
    </>
  );

  const renderInsights = () => (
    <>
      <section className="welcome">
        <div>
          <p className="eyebrow light">MODEL EVALUATION</p>
          <h2>Understand model performance.</h2>
          <p>Compare the trained Random Forest against the baseline and Linear Regression experiments.</p>
        </div>
        <div className="welcome-symbol">⌁</div>
      </section>

      <section className="stats-grid">
        <article className="stat-card"><div className="stat-top"><span>Selected Model</span><span className="stat-icon green">✓</span></div><h2 className="text-stat">Random Forest</h2><p>Saved model used by the API</p></article>
        <article className="stat-card"><div className="stat-top"><span>Random Forest MAE</span><span className="stat-icon blue">↓</span></div><h2>5.0579</h2><p>Mean Absolute Error · beds</p></article>
        <article className="stat-card"><div className="stat-top"><span>Random Forest RMSE</span><span className="stat-icon purple">∑</span></div><h2>11.0024</h2><p>Root Mean Squared Error · beds</p></article>
        <article className="stat-card"><div className="stat-top"><span>Random Forest R²</span><span className="stat-icon orange">R</span></div><h2>0.9962</h2><p>Coefficient of determination</p></article>
      </section>

      <section className="charts-grid">
        <article className="panel chart-panel">
          <p className="eyebrow">ERROR COMPARISON</p>
          <h2>MAE and RMSE by model</h2>
          <p className="panel-subtitle">Lower error values are generally better.</p>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelResults}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="model" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="MAE" fill="#167e78" radius={[5, 5, 0, 0]} />
                <Bar dataKey="RMSE" fill="#8b78c8" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel chart-panel">
          <p className="eyebrow">MODEL FIT</p>
          <h2>R² score comparison</h2>
          <p className="panel-subtitle">Reported coefficient of determination.</p>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelResults}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="model" tick={{ fontSize: 10 }} />
                <YAxis domain={[0.99, 1]} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="R2" name="R² score" fill="#e3a53d" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="panel history-panel">
        <p className="eyebrow">EXPERIMENT RESULTS</p>
        <h2>Model comparison</h2>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr></thead>
            <tbody>
              {modelResults.map((item) => (
                <tr key={item.model}>
                  <td>{item.model} {item.model === "Random Forest" && <span className="table-tag">Selected</span>}</td>
                  <td>{item.MAE.toFixed(4)}</td><td>{item.RMSE.toFixed(4)}</td><td>{item.R2.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="input-help metrics-note">Metrics are from the recorded model-comparison experiment. They are not recalculated by this dashboard.</p>
      </section>
    </>
  );

  const renderMonitoring = () => (
    <>
      <section className="welcome">
        <div>
          <p className="eyebrow light">SERVICE OBSERVABILITY</p>
          <h2>Monitor your prediction API.</h2>
          <p>Check backend availability and inspect the Prometheus metrics exposed by the Flask service.</p>
        </div>
        <div className="welcome-symbol">◉</div>
      </section>

      <section className="monitor-toolbar">
        <div><strong>Backend endpoint</strong><p>{API}</p></div>
        <button className="secondary-button" onClick={checkBackend} disabled={healthLoading}>
          {healthLoading ? "Checking..." : "↻ Refresh status"}
        </button>
      </section>

      <section className="stats-grid">
        <article className="stat-card">
          <div className="stat-top"><span>API Health</span><span className="stat-icon green">♥</span></div>
          <h2 className="text-stat">{health ? "Reachable" : "Unknown"}</h2>
          <p>{health ? "Health endpoint responded" : "Refresh to check the backend"}</p>
        </article>
        <article className="stat-card">
          <div className="stat-top"><span>Prometheus Metrics</span><span className="stat-icon purple">▤</span></div>
          <h2 className="text-stat">{metricsText ? "Available" : "Not loaded"}</h2>
          <p>Metrics endpoint response</p>
        </article>
        <article className="stat-card">
          <div className="stat-top"><span>Prediction API</span><span className="stat-icon blue">↗</span></div>
          <h2 className="text-stat">/predict</h2>
          <p>POST · JSON input</p>
        </article>
        <article className="stat-card">
          <div className="stat-top"><span>Predictions this session</span><span className="stat-icon orange">∑</span></div>
          <h2>{history.length}</h2>
          <p>Generated in this browser session</p>
        </article>
      </section>

      <section className="content-grid">
        <article className="panel">
          <p className="eyebrow">HEALTH CHECK RESPONSE</p>
          <h2>Backend status</h2>
          {healthError ? <div className="message error-message">{healthError}</div> :
            health ? <pre className="response-code">{JSON.stringify(health, null, 2)}</pre> :
            <p className="panel-subtitle">Click Refresh status to check the API.</p>}
          {metricsError && <div className="message error-message">{metricsError}</div>}
        </article>

        <article className="panel">
          <p className="eyebrow">PROMETHEUS</p>
          <h2>Metrics endpoint</h2>
          <p className="panel-subtitle">Raw monitoring metrics exposed by Flask.</p>
          <a className="text-link" href={`${API}/metrics`} target="_blank" rel="noreferrer">Open raw /metrics endpoint ↗</a>
          <div className="metric-list">
            {metricLines.length > 0 ? metricLines.map((line, index) => (
              <code key={index}>{line}</code>
            )) : <p className="panel-subtitle">Metrics will appear here after a successful refresh.</p>}
          </div>
        </article>
      </section>
    </>
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">H</div>
          <div><h2>HospitaLens</h2><span>CAPACITY INTELLIGENCE</span></div>
        </div>

        <p className="nav-label">WORKSPACE</p>
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${page === item.id ? "active" : ""}`}
            onClick={() => setPage(item.id)}
          >
            <span>{item.icon}</span>{item.label}
          </button>
        ))}

        <div className="sidebar-bottom">
          <div className={`status-dot ${health ? "online" : ""}`} />
          <div><strong>Prediction service</strong><p>{health ? "Health check successful" : "Check status in Monitoring"}</p></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">{pageTitles[page][0]}</p>
            <h1>{pageTitles[page][1]}</h1>
          </div>
          <div className="topbar-right">
            <span className="environment-badge">ML MODEL · RANDOM FOREST</span>
            <div className="avatar">HL</div>
          </div>
        </header>

        {page === "overview" && renderOverview()}
        {page === "forecast" && renderForecastForm()}
        {page === "insights" && renderInsights()}
        {page === "monitoring" && renderMonitoring()}

        <footer>
          <span>HospitaLens · Hospital Bed Occupancy Prediction</span>
          <span>Academic MLOps Project · Experimental estimates</span>
        </footer>
      </main>
    </div>
  );
}

function HistoryTable({ history }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Time</th><th>Current occupancy</th><th>Predicted next week</th></tr></thead>
        <tbody>
          {history.map((item) => (
            <tr key={item.id}>
              <td>{item.date}</td><td>{item.current}</td><td>{item.predicted.toFixed(2)} beds</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;