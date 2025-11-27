document.addEventListener("DOMContentLoaded", () => {
  const searchBtn = document.getElementById("search-btn");
  const queryEl = document.getElementById("query");

  if (searchBtn) {
    searchBtn.onclick = () => {
      const q = queryEl.value.trim();
      if (!q) return;
      // Redirect to results page with query
      window.location.href = `/static/results.html?q=${encodeURIComponent(q)}`;
    };
  }

  // On results page, fetch and display
  const resultsContainer = document.getElementById("results");
  if (resultsContainer) {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    if (q) {
      queryEl.value = q;
      resultsContainer.innerHTML = `<p>Searching for "${q}"...</p>`;
      fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, k: 5 })
      })
        .then(resp => resp.json())
        .then(data => {
          resultsContainer.innerHTML = "";
          data.retrieved.forEach(r => {
            const item = document.createElement("div");
            item.className = "result-item";
            item.innerHTML = `
              <h3>${r.meta?.source || "Unknown Source"}</h3>
              <p>${r.text.slice(0,200)}...</p>
              <small>Score: ${(r.final_score||0).toFixed(3)}</small>
            `;
            resultsContainer.appendChild(item);
          });
          const answer = document.createElement("div");
          answer.className = "result-item";
          answer.innerHTML = `<h3>Answer</h3><p>${data.answer}</p>`;
          resultsContainer.prepend(answer);
        })
        .catch(err => {
          resultsContainer.innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
        });
    }
  }
});