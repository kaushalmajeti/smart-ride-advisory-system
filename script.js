const MAPBOX_KEY = "yor-mapbox-key";
const BACKEND_URL = "http://127.0.0.1:5000";
const AUTO_MAX_KM = 15;

let currentRideData = null;
const autocompleteTimers = {};

function escapeHTML(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatMoney(value) {
  return `Rs. ${Number(value).toFixed(2)}`;
}

function rideCode(name) {
  return {
    "Ola Auto": "AUTO",
    "Ola Mini": "MINI",
    "Uber Go": "GO",
    "Uber Sedan": "SEDAN",
  }[name] || "RIDE";
}

function modelLabel(modelUsed) {
  return {
    stacking_ensemble: "Stacking + LSTM",
    stacking_only: "Stacking only",
  }[modelUsed] || String(modelUsed || "Model");
}

async function handleInput(input, listId) {
  const query = input.value.trim();
  const list = document.getElementById(listId);
  if (!list) return;

  clearTimeout(autocompleteTimers[listId]);
  if (query.length < 3) {
    list.innerHTML = "";
    return;
  }

  autocompleteTimers[listId] = setTimeout(async () => {
    try {
      const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json?access_token=${MAPBOX_KEY}&limit=5&country=in`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Location search failed");
      const data = await res.json();

      list.innerHTML = "";
      (data.features || []).forEach((place) => {
        const div = document.createElement("div");
        div.innerText = place.place_name;
        div.onclick = () => {
          input.value = place.place_name;
          list.innerHTML = "";
        };
        list.appendChild(div);
      });
    } catch (e) {
      console.error("Autocomplete error:", e);
      list.innerHTML = "";
    }
  }, 250);
}

async function getCoordinates(place) {
  const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(place)}.json?access_token=${MAPBOX_KEY}&country=in`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Could not search for location.");
  const data = await res.json();
  if (!data.features || data.features.length === 0) {
    throw new Error(`Location not found: ${place}`);
  }
  return data.features[0].center;
}

function calculateFare(distance, surgeMultiplier = 1.0) {
  return [
    { name: "Ola Auto", price: (20 + distance * 6) * surgeMultiplier, basePrice: 20 + distance * 6, unavailable: distance > AUTO_MAX_KM },
    { name: "Ola Mini", price: (40 + distance * 8) * surgeMultiplier, basePrice: 40 + distance * 8, unavailable: false },
    { name: "Uber Go", price: (30 + distance * 7) * surgeMultiplier, basePrice: 30 + distance * 7, unavailable: false },
    { name: "Uber Sedan", price: (60 + distance * 12) * surgeMultiplier, basePrice: 60 + distance * 12, unavailable: false },
  ];
}

function suggestRide(distance) {
  if (distance <= AUTO_MAX_KM && distance < 3) return "Ola Auto";
  if (distance < 8) return "Ola Mini";
  if (distance < 20) return "Uber Go";
  return "Uber Sedan";
}

function getSegmentCode(segment) {
  return {
    "New User": "NEW",
    "Budget Rider": "SAVE",
    "Premium Commuter": "PRO",
    "Weekend Explorer": "TRIP",
    "Night Owl": "NIGHT",
    "Regular Rider": "RIDE",
  }[segment] || "RIDE";
}

async function loadUserProfile() {
  const user = localStorage.getItem("user");
  const container = document.getElementById("user-profile");
  if (!user || !container) return;

  try {
    const res = await fetch(`${BACKEND_URL}/userProfile/${encodeURIComponent(user)}`);
    if (!res.ok) {
      container.innerHTML = "";
      return;
    }
    const p = await res.json();

    if (!p.total_rides) {
      container.innerHTML = `
        <div class="profile-card profile-new">
          <span class="profile-icon profile-code">NEW</span>
          <div class="profile-info">
            <span class="profile-segment">New User</span>
            <span class="profile-sub">Book your first ride to unlock personal recommendations.</span>
          </div>
        </div>`;
      return;
    }

    const rideDistHTML = Object.entries(p.ride_distribution || {})
      .sort((a, b) => b[1] - a[1])
      .map(([name, pct]) => `
        <div class="profile-bar-row">
          <span class="profile-bar-label">${escapeHTML(name)}</span>
          <div class="profile-bar-track">
            <div class="profile-bar-fill" style="width:${Number(pct)}%"></div>
          </div>
          <span class="profile-bar-pct">${Number(pct).toFixed(1)}%</span>
        </div>`)
      .join("");

    container.innerHTML = `
      <div class="profile-card">
        <div class="profile-header">
          <div class="profile-left">
            <span class="profile-icon profile-code">${getSegmentCode(p.segment)}</span>
            <div>
              <span class="profile-segment">${escapeHTML(p.segment)}</span>
              <span class="profile-tier">${escapeHTML(p.spending_tier)} Tier</span>
            </div>
          </div>
          <div class="profile-badges">
            ${p.peak_rider ? '<span class="pbadge pbadge-peak">Peak Rider</span>' : ""}
            ${p.night_rider ? '<span class="pbadge pbadge-night">Night Rider</span>' : ""}
          </div>
        </div>
        <div class="profile-stats">
          <div class="pstat"><span class="pstat-val">${p.total_rides}</span><span class="pstat-lbl">Rides</span></div>
          <div class="pstat"><span class="pstat-val">${p.avg_distance} km</span><span class="pstat-lbl">Avg Distance</span></div>
          <div class="pstat"><span class="pstat-val">Rs. ${p.avg_price}</span><span class="pstat-lbl">Avg Fare</span></div>
          <div class="pstat"><span class="pstat-val">${p.weekend_ratio}%</span><span class="pstat-lbl">Weekend</span></div>
        </div>
        <div class="profile-dist">
          <span class="profile-dist-title">Ride Preference</span>
          ${rideDistHTML || '<span class="profile-sub">No preference data yet.</span>'}
        </div>
      </div>`;
  } catch (e) {
    console.warn("Could not load profile:", e);
    container.innerHTML = "";
  }
}

async function fetchMLRecommendation(distance) {
  const user = localStorage.getItem("user");
  if (!user) return null;

  try {
    const res = await fetch(`${BACKEND_URL}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId: user, distance }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    console.warn("ML recommend unavailable:", e);
    return null;
  }
}

function renderSurgeIndicator(surge) {
  if (!surge || surge.multiplier <= 1.05) return "";
  const levelClass = `surge-${escapeHTML(surge.level)}`;
  return `
    <div class="surge-card ${levelClass}">
      <div class="surge-header">
        <span class="surge-icon">SURGE</span>
        <span class="surge-title">Dynamic pricing active</span>
        <span class="surge-mult">${Number(surge.multiplier).toFixed(2)}x</span>
      </div>
      <div class="surge-reason">${escapeHTML(surge.reason)}</div>
    </div>`;
}

function renderSHAPExplanation(shap) {
  if (!shap || Object.keys(shap).length === 0) return "";

  const maxVal = Math.max(...Object.values(shap).map((value) => Math.abs(Number(value))));
  const bars = Object.entries(shap)
    .slice(0, 5)
    .map(([feat, val]) => {
      const numericVal = Number(val);
      const pct = Math.round((Math.abs(numericVal) / (maxVal || 1)) * 100);
      const dir = numericVal >= 0 ? "positive" : "negative";
      return `
        <div class="shap-row">
          <span class="shap-feat">${escapeHTML(feat)}</span>
          <div class="shap-bar-track">
            <div class="shap-bar-fill shap-${dir}" style="width:${pct}%"></div>
          </div>
          <span class="shap-val ${dir === "positive" ? "shap-pos" : "shap-neg"}">${numericVal >= 0 ? "+" : ""}${numericVal.toFixed(3)}</span>
        </div>`;
    })
    .join("");

  return `
    <div class="shap-section">
      <span class="shap-title">Why this recommendation?</span>
      <div class="shap-bars">${bars}</div>
    </div>`;
}

function renderMLCard(ml, fares) {
  if (!ml || !ml.recommended) return "";

  const ride = fares.find((f) => f.name === ml.recommended);
  const pct = Math.round((Number(ml.confidence) || 0) * 100);
  const bars = Object.entries(ml.scores || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, score]) => {
      const barPct = Math.round(Number(score) * 100);
      const isTop = name === ml.recommended;
      return `
        <div class="ml-bar-row">
          <span class="ml-bar-label">${rideCode(name)} ${escapeHTML(name)}</span>
          <div class="ml-bar-track">
            <div class="ml-bar-fill ${isTop ? "ml-bar-top" : ""}" style="width:${barPct}%"></div>
          </div>
          <span class="ml-bar-pct">${barPct}%</span>
        </div>`;
    })
    .join("");

  return `
    <div class="ml-card">
      <div class="ml-header">
        <span class="ml-title">AI Recommendation</span>
        <div class="ml-tags">
          <span class="ml-tag">${escapeHTML(modelLabel(ml.model_used))}</span>
          ${ml.user_segment ? `<span class="ml-segment">${getSegmentCode(ml.user_segment)} ${escapeHTML(ml.user_segment)}</span>` : ""}
        </div>
      </div>
      <div class="ml-main">
        <span class="ml-ride-icon ride-code">${rideCode(ride ? ride.name : ml.recommended)}</span>
        <div>
          <div class="ml-ride-name">${escapeHTML(ml.recommended)}</div>
          <div class="ml-confidence">${pct}% confidence</div>
        </div>
      </div>
      <div class="ml-bars">${bars}</div>
      ${ml.segment_reason ? `<p class="ml-reason">${escapeHTML(ml.segment_reason)}</p>` : ""}
      ${renderSHAPExplanation(ml.shap_explanation)}
    </div>`;
}

function renderTripSummary(distance, fares, suggestion) {
  const available = fares.filter((fare) => !fare.unavailable);
  const cheapest = [...available].sort((a, b) => a.price - b.price)[0];
  const highest = [...available].sort((a, b) => b.price - a.price)[0];
  const etaMinutes = Math.max(6, Math.round((distance / 24) * 60 + 5));
  const savings = highest && cheapest ? Math.max(0, highest.price - cheapest.price) : 0;

  return `
    <div class="trip-summary">
      <div class="summary-item">
        <span class="summary-label">Route</span>
        <strong>${distance.toFixed(2)} km</strong>
      </div>
      <div class="summary-item">
        <span class="summary-label">Best value</span>
        <strong>${escapeHTML(suggestion)}</strong>
      </div>
      <div class="summary-item">
        <span class="summary-label">Cheapest</span>
        <strong>${escapeHTML(cheapest.name)} (${formatMoney(cheapest.price)})</strong>
      </div>
      <div class="summary-item">
        <span class="summary-label">Est. travel time</span>
        <strong>${etaMinutes} min</strong>
      </div>
      <div class="summary-item">
        <span class="summary-label">Possible savings</span>
        <strong>${formatMoney(savings)}</strong>
      </div>
    </div>`;
}

function renderCards(fares, distance, selectedName, ml) {
  const suggestion = ml ? ml.recommended : suggestRide(distance);
  const autoNote = distance > AUTO_MAX_KM
    ? `<span class="tag-unavail">Auto unavailable beyond ${AUTO_MAX_KM} km</span>`
    : "";
  const surgeHTML = ml && ml.surge ? renderSurgeIndicator(ml.surge) : "";
  const surgeMultiplier = ml && ml.surge ? Number(ml.surge.multiplier) : 1.0;

  return `
    ${renderTripSummary(distance, fares, suggestion)}
    ${surgeHTML}
    ${renderMLCard(ml, fares)}
    <p class="suggestion">Recommended: <strong>${escapeHTML(suggestion)}</strong> ${autoNote}</p>
    <div class="cards">
      ${fares.map((f) => {
        const isSel = f.name === selectedName;
        const isUna = f.unavailable;
        const isRec = f.name === suggestion;
        const hasSurge = surgeMultiplier > 1.05;
        return `
          <div class="card ${isSel ? "selected" : ""} ${isUna ? "unavailable" : ""}"
               onclick="${isUna ? "" : `selectRide('${f.name}')`}">
            <div class="card-icon ride-code">${rideCode(f.name)}</div>
            <h4>${escapeHTML(f.name)}</h4>
            <p class="card-price">${formatMoney(f.price)}</p>
            ${hasSurge && !isUna ? `<p class="card-base-price">Base: ${formatMoney(f.basePrice)}</p>` : ""}
            ${isRec && !isUna ? '<span class="badge badge-rec">Recommended</span>' : ""}
            ${isUna ? '<span class="badge badge-unavail">Not available</span>' : ""}
            ${isSel ? '<span class="badge badge-sel">Selected</span>' : ""}
            ${hasSurge && !isUna ? `<span class="badge badge-surge">${surgeMultiplier.toFixed(2)}x surge</span>` : ""}
          </div>`;
      }).join("")}
    </div>
    ${selectedName
      ? (() => {
          const chosen = fares.find((f) => f.name === selectedName);
          return `
            <div class="book-row">
              <div class="selected-summary">
                <strong>${escapeHTML(chosen.name)}</strong> - ${formatMoney(chosen.price)}
              </div>
              <button class="btn-book" onclick="bookRide()">Book Now</button>
            </div>`;
        })()
      : '<p class="pick-hint">Select a ride to continue.</p>'
    }`;
}

function selectRide(name) {
  if (!currentRideData) return;
  currentRideData.selectedRide = name;
  document.getElementById("result").innerHTML =
    renderCards(currentRideData.fares, currentRideData.distance, name, currentRideData.ml);
}

async function bookRide() {
  if (!currentRideData || !currentRideData.selectedRide) return;

  const { fares, distance, selectedRide, pickupInput, dropInput } = currentRideData;
  const chosen = fares.find((f) => f.name === selectedRide);
  const user = localStorage.getItem("user");

  document.getElementById("result").innerHTML = `
    <div class="booked-box">
      <div class="booked-icon ride-code">DONE</div>
      <h3>Ride booked</h3>
      <p class="booked-ride"><strong>${escapeHTML(chosen.name)}</strong></p>
      <p class="booked-detail">${escapeHTML(pickupInput)} to ${escapeHTML(dropInput)}</p>
      <p class="booked-price">${formatMoney(chosen.price)} | ${distance.toFixed(2)} km</p>
      <button class="btn-new" onclick="newSearch()">Book Another Ride</button>
    </div>`;

  if (user) {
    try {
      await fetch(`${BACKEND_URL}/saveRide`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: user,
          pickup: pickupInput,
          drop: dropInput,
          distance: parseFloat(distance.toFixed(2)),
          ride: chosen.name,
          price: parseFloat(chosen.price.toFixed(2)),
        }),
      });
      loadHistory();
      loadUserProfile();
    } catch (e) {
      console.warn("Could not save ride:", e);
    }
  }
  currentRideData = null;
}

function newSearch() {
  document.getElementById("pickup").value = "";
  document.getElementById("drop").value = "";
  document.getElementById("pickupList").innerHTML = "";
  document.getElementById("dropList").innerHTML = "";
  document.getElementById("result").innerHTML = "";
  currentRideData = null;
}

async function getDistance() {
  const pickupInput = document.getElementById("pickup").value.trim();
  const dropInput = document.getElementById("drop").value.trim();
  const result = document.getElementById("result");
  const button = document.querySelector(".btn-check");

  if (!pickupInput || !dropInput) {
    result.innerHTML = '<p class="error-msg">Enter both pickup and drop locations.</p>';
    return;
  }

  button.disabled = true;
  button.innerText = "Calculating...";
  result.innerHTML = '<p class="loading">Calculating route, fares, and AI recommendation...</p>';

  try {
    const pickup = await getCoordinates(pickupInput);
    const drop = await getCoordinates(dropInput);

    const url = `https://api.mapbox.com/directions/v5/mapbox/driving/${pickup[0]},${pickup[1]};${drop[0]},${drop[1]}?access_token=${MAPBOX_KEY}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not calculate route.");
    const data = await res.json();

    if (!data.routes || data.routes.length === 0) {
      result.innerHTML = '<p class="error-msg">No route found between these locations.</p>';
      return;
    }

    const distance = data.routes[0].distance / 1000;
    const ml = await fetchMLRecommendation(distance);
    const surgeMultiplier = ml && ml.surge ? Number(ml.surge.multiplier) : 1.0;
    const fares = calculateFare(distance, surgeMultiplier);

    currentRideData = { fares, distance, pickupInput, dropInput, selectedRide: null, ml };
    result.innerHTML = renderCards(fares, distance, null, ml);
  } catch (error) {
    console.error(error);
    result.innerHTML = `<p class="error-msg">${escapeHTML(error.message)}</p>`;
  } finally {
    button.disabled = false;
    button.innerText = "Check Prices";
  }
}

async function loadHistory() {
  const user = localStorage.getItem("user");
  const container = document.getElementById("history");
  if (!user || !container) return;

  try {
    const res = await fetch(`${BACKEND_URL}/getHistory/${encodeURIComponent(user)}`);
    if (!res.ok) throw new Error("Could not load history.");
    const payload = await res.json();
    const data = Array.isArray(payload) ? payload : payload.items || [];

    if (!data.length) {
      container.innerHTML = "<p class='no-history'>No ride history yet.</p>";
      return;
    }

    data.sort((a, b) => Number(b.timestamp) - Number(a.timestamp));
    container.innerHTML = `
      <div class="history-head">
        <h3 class="history-title">Recent Rides</h3>
        ${payload.storage ? `<span class="storage-pill">${escapeHTML(payload.storage)} storage</span>` : ""}
      </div>
      <table>
        <thead><tr><th>From</th><th>To</th><th>Distance</th><th>Ride</th><th>Price</th></tr></thead>
        <tbody>
          ${data.map((r) => `
            <tr>
              <td>${escapeHTML(r.pickup)}</td>
              <td>${escapeHTML(r.drop)}</td>
              <td>${Number(r.distance).toFixed(2)} km</td>
              <td>${escapeHTML(r.chosenRide)}</td>
              <td>${formatMoney(r.price)}</td>
            </tr>`).join("")}
        </tbody>
      </table>`;
  } catch (e) {
    console.warn("Could not load history:", e);
    container.innerHTML = "<p class='no-history'>Could not load history. Start the backend and try again.</p>";
  }
}

let metricsLoaded = false;

async function toggleMetrics() {
  const content = document.getElementById("metrics-content");
  if (content.style.display === "none") {
    content.style.display = "block";
    if (!metricsLoaded) {
      content.innerHTML = '<p class="loading">Loading model metrics...</p>';
      await loadMetrics();
      metricsLoaded = true;
    }
  } else {
    content.style.display = "none";
  }
}

async function loadMetrics() {
  const container = document.getElementById("metrics-content");
  try {
    const res = await fetch(`${BACKEND_URL}/modelMetrics`);
    if (!res.ok) throw new Error("Failed to fetch metrics");
    const m = await res.json();

    const renderModelCard = (data, accent) => {
      if (!data || !data.overall_accuracy) return "";
      const cm = data.confusion_matrix;
      const cmHTML = cm ? `
        <div class="cm-grid">
          ${cm.map((row, i) => row.map((val, j) =>
            `<div class="cm-cell ${i === j ? "cm-diag" : ""}">${val}</div>`
          ).join("")).join("")}
        </div>` : "";

      return `
        <div class="metric-card" style="border-color:${accent}">
          <div class="metric-header">
            <span class="metric-name">${escapeHTML(data.model)}</span>
            <span class="metric-acc" style="color:${accent}">${data.overall_accuracy}%</span>
          </div>
          <p class="metric-purpose">${escapeHTML(data.purpose)}</p>
          ${data.architecture ? `<p class="metric-arch">Architecture: ${escapeHTML(data.architecture)}</p>` : ""}
          <div class="metric-stats">
            <span>F1: ${data.macro_f1}%</span>
            ${data.cross_val_mean ? `<span>CV: ${data.cross_val_mean}+/-${data.cross_val_std}%</span>` : ""}
            <span>Train: ${data.train_size}</span>
            <span>Test: ${data.test_size || "N/A"}</span>
          </div>
          ${cm ? `<div class="cm-section"><span class="cm-title">Confusion Matrix</span>${cmHTML}</div>` : ""}
        </div>`;
    };

    const renderSurgeCard = (data) => {
      if (!data) return "";
      return `
        <div class="metric-card" style="border-color:#f0c040">
          <div class="metric-header">
            <span class="metric-name">${escapeHTML(data.model)}</span>
            <span class="metric-acc" style="color:#f0c040">R2=${data.r2}</span>
          </div>
          <p class="metric-purpose">${escapeHTML(data.purpose)}</p>
          <div class="metric-stats">
            <span>MAE: ${data.mae}</span>
            <span>RMSE: ${data.rmse}</span>
            <span>Train: ${data.train_size}</span>
          </div>
        </div>`;
    };

    const renderClusterCard = (data) => {
      if (!data) return "";
      const centroids = Object.entries(data.centroids || {}).map(([name, c]) => `
        <div class="cluster-row">
          <span class="cluster-name">${escapeHTML(name)}</span>
          <span class="cluster-detail">~${c.avg_distance}km, ${Number(c.peak_ratio) > 0.3 ? "Peak" : Number(c.night_ratio) > 0.2 ? "Night" : Number(c.weekend_ratio) > 0.3 ? "Weekend" : "Budget"}</span>
        </div>`).join("");
      return `
        <div class="metric-card" style="border-color:#4cde9e">
          <div class="metric-header">
            <span class="metric-name">${escapeHTML(data.model)}</span>
            <span class="metric-acc" style="color:#4cde9e">${data.n_clusters} segments</span>
          </div>
          <p class="metric-purpose">${escapeHTML(data.purpose)}</p>
          <div class="cluster-list">${centroids}</div>
        </div>`;
    };

    container.innerHTML = `
      <div class="metrics-grid">
        <div class="metrics-col">
          <h4 class="metrics-section-title">Stacking Ensemble</h4>
          ${renderModelCard(m.stacking_ensemble, "#f0c040")}
          <h4 class="metrics-section-title">Base Models</h4>
          ${Object.values(m.base_models || {}).map((bm) => renderModelCard(bm, "#7a7885")).join("")}
        </div>
        <div class="metrics-col">
          <h4 class="metrics-section-title">LSTM Sequence Model</h4>
          ${renderModelCard(m.lstm, "#a78bfa")}
          <h4 class="metrics-section-title">Surge Pricing</h4>
          ${renderSurgeCard(m.surge_pricing)}
          <h4 class="metrics-section-title">User Clustering</h4>
          ${renderClusterCard(m.user_clustering)}
        </div>
      </div>
      <div class="metrics-footer">
        <span>Ensemble Weights: Stacking ${(m.ensemble_weights?.stacking || 0.55) * 100}% + LSTM ${(m.ensemble_weights?.lstm || 0.45) * 100}%</span>
        <span>Total ML Models: ${m.total_models || 6}</span>
      </div>`;
  } catch (e) {
    console.error("Metrics error:", e);
    container.innerHTML = '<p class="error-msg">Could not load model metrics. Is the backend running?</p>';
  }
}

function logout() {
  localStorage.removeItem("user");
  localStorage.removeItem("idToken");
  window.location.href = "login.html";
}

window.onload = () => {
  loadHistory();
  loadUserProfile();
};
