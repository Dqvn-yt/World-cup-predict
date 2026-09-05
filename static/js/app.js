const toggle = document.querySelector('[data-nav-toggle]');
const nav = document.querySelector('[data-nav]');
if (toggle && nav) toggle.addEventListener('click', () => nav.classList.toggle('open'));

window.renderFormChart = (id, labels, values) => {
  const canvas = document.getElementById(id);
  if (!canvas || typeof Chart === 'undefined') return;
  new Chart(canvas, {
    type: 'line',
    data: { labels, datasets: [{ data: values, borderColor: '#b9ff36', backgroundColor: 'rgba(185,255,54,.12)', fill: true, tension: .32, pointRadius: 4 }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 10, grid: { color: '#24312e' }, ticks: { color: '#81918c' } }, x: { grid: { display: false }, ticks: { color: '#81918c' } } } }
  });
};
