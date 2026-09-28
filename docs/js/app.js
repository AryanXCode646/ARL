/**
 * ARL - Adaptive Reinforcement Learning
 * Main Website Application Logic (Charts, Copy, Accordion, Mobile Nav)
 */

(function () {
  'use strict';

  function init() {
    initCopyButtons();
    initAccordions();
    initMobileNav();
    initCharts();
    initScrollSpy();
  }

  function initCopyButtons() {
    const copyBtns = document.querySelectorAll('.copy-btn');
    copyBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-target');
        const codeEl = document.getElementById(targetId);
        if (!codeEl) return;

        const text = codeEl.innerText.trim();
        navigator.clipboard.writeText(text).then(() => {
          const originalText = btn.textContent;
          btn.textContent = 'Copied!';
          btn.style.color = '#10b981';
          setTimeout(() => {
            btn.textContent = originalText;
            btn.style.color = '';
          }, 2000);
        }).catch(err => {
          console.error('Clipboard copy failed:', err);
        });
      });
    });
  }

  function initAccordions() {
    const headers = document.querySelectorAll('.accordion-header');
    headers.forEach(h => {
      h.addEventListener('click', () => {
        const targetId = h.getAttribute('data-accordion');
        const content = document.getElementById(targetId);
        if (!content) return;

        const isOpen = content.classList.contains('active');
        content.classList.toggle('active', !isOpen);
        const icon = h.querySelector('.accordion-icon');
        if (icon) icon.textContent = isOpen ? '+' : '−';
      });
    });
  }

  function initMobileNav() {
    const menuBtn = document.getElementById('mobile-menu-btn');
    const navLinks = document.getElementById('nav-links');
    if (!menuBtn || !navLinks) return;

    menuBtn.addEventListener('click', () => {
      navLinks.classList.toggle('active');
    });

    const links = navLinks.querySelectorAll('a');
    links.forEach(l => {
      l.addEventListener('click', () => {
        navLinks.classList.remove('active');
      });
    });
  }

  function initCharts() {
    if (typeof Chart === 'undefined') return;

    // Set dark theme defaults for Chart.js
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';

    // 1. PPO vs Random Baseline Chart
    const ppoCtx = document.getElementById('ppo-vs-random-chart');
    if (ppoCtx) {
      new Chart(ppoCtx, {
        type: 'bar',
        data: {
          labels: ['Success Rate (%)', 'Collision Rate (%)', 'Survival Rate (%)'],
          datasets: [
            {
              label: 'Random Policy (Baseline)',
              data: [0.0, 100.0, 0.0],
              backgroundColor: 'rgba(239, 68, 68, 0.65)',
              borderColor: '#ef4444',
              borderWidth: 1.5,
              borderRadius: 4
            },
            {
              label: 'Trained PPO Agent (Demo)',
              data: [5.0, 35.0, 65.0],
              backgroundColor: 'rgba(0, 242, 254, 0.75)',
              borderColor: '#00f2fe',
              borderWidth: 1.5,
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top',
              labels: { boxWidth: 14, font: { size: 12 } }
            },
            tooltip: {
              callbacks: {
                label: function (context) {
                  return ` ${context.dataset.label}: ${context.raw}%`;
                }
              }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: {
                callback: function (val) { return val + '%'; }
              }
            },
            x: {
              grid: { display: false }
            }
          }
        }
      });
    }

    // 2. Obstacle Density Scaling Chart
    const densityCtx = document.getElementById('density-scaling-chart');
    if (densityCtx) {
      new Chart(densityCtx, {
        type: 'line',
        data: {
          labels: ['Low (4 Obstacles)', 'Medium (6 Obstacles)', 'High (8 Obstacles)'],
          datasets: [
            {
              label: 'Collision Rate (%)',
              data: [20.0, 40.0, 70.0],
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              fill: true,
              tension: 0.3,
              pointRadius: 6,
              pointHoverRadius: 8,
              yAxisID: 'y'
            },
            {
              label: 'Mean Episodic Return',
              data: [5.0, -15.2, -32.43],
              borderColor: '#10b981',
              backgroundColor: 'transparent',
              borderDash: [5, 5],
              tension: 0.3,
              pointRadius: 6,
              pointHoverRadius: 8,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top',
              labels: { boxWidth: 14, font: { size: 12 } }
            }
          },
          scales: {
            y: {
              type: 'linear',
              position: 'left',
              title: { display: true, text: 'Collision Rate (%)', color: '#ef4444' },
              beginAtZero: true,
              max: 100,
              grid: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            y1: {
              type: 'linear',
              position: 'right',
              title: { display: true, text: 'Mean Return', color: '#10b981' },
              grid: { drawOnChartArea: false }
            },
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' }
            }
          }
        }
      });
    }
  }

  function initScrollSpy() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', () => {
      let current = '';
      const scrollPos = window.scrollY + 200;

      sections.forEach(s => {
        const top = s.offsetTop;
        const height = s.offsetHeight;
        if (scrollPos >= top && scrollPos < top + height) {
          current = s.getAttribute('id');
        }
      });

      navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
          link.classList.add('active');
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
