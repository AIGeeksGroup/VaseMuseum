(function () {
  var copyBtn = document.getElementById('copy-bibtex');
  var bibtexEl = document.getElementById('bibtex');
  if (copyBtn && bibtexEl) {
    copyBtn.addEventListener('click', function () {
      var text = bibtexEl.innerText;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          copyBtn.textContent = 'Copied!';
          setTimeout(function () {
            copyBtn.textContent = 'Copy BibTeX';
          }, 1500);
        }).catch(function () {
          fallbackCopy(text);
        });
      } else {
        fallbackCopy(text);
      }
    });
  }
  function fallbackCopy(str) {
    var ta = document.createElement('textarea');
    ta.value = str;
    ta.setAttribute('readonly', '');
    ta.style.position = 'absolute';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      if (copyBtn) copyBtn.textContent = 'Copied!';
      setTimeout(function () {
        if (copyBtn) copyBtn.textContent = 'Copy BibTeX';
      }, 1500);
    } catch (e) {}
    document.body.removeChild(ta);
  }
})();

(function () {
  var astronautFallbacks = [
    'https://modelviewer.dev/shared-assets/models/Astronaut.glb',
    'https://cdn.jsdelivr.net/gh/google/model-viewer/packages/shared-assets/models/Astronaut.glb',
    'https://raw.githubusercontent.com/google/model-viewer/master/packages/shared-assets/models/Astronaut.glb'
  ];
  var astronautViewers = document.querySelectorAll('model-viewer[src*="Astronaut.glb"]');
  if (!astronautViewers.length) return;

  function showModelErrorHint(viewer) {
    var card = viewer.closest('.mesh-card') || viewer.parentElement;
    if (!card || card.querySelector('.mesh-model-error')) return;
    var hint = document.createElement('p');
    hint.className = 'mesh-model-error';
    hint.textContent = '3D model failed to load. Check your network and try again.';
    card.appendChild(hint);
  }

  astronautViewers.forEach(function (viewer) {
    viewer.dataset.fallbackIndex = '0';
    viewer.addEventListener('error', function () {
      var currentIndex = parseInt(viewer.dataset.fallbackIndex || '0', 10);
      var nextIndex = currentIndex + 1;
      if (nextIndex < astronautFallbacks.length) {
        viewer.dataset.fallbackIndex = String(nextIndex);
        viewer.src = astronautFallbacks[nextIndex];
        return;
      }
      showModelErrorHint(viewer);
    });
  });
})();

(function () {
  var exhibitionFrame = document.getElementById('exhibition-iframe');
  if (!exhibitionFrame) return;
  var exhibitionWrap = exhibitionFrame.closest('.exhibition-wrap');
  const DEFAULT_ROTATION_SPEED = 80;

  function postToExhibition(cmd) {
    if (!exhibitionFrame || !exhibitionFrame.contentWindow) return false;
    exhibitionFrame.contentWindow.postMessage(
      Object.assign({ source: 'vase-exhibition-controls' }, cmd),
      '*'
    );
    return true;
  }

  function setIframeInteractionActive(active) {
    postToExhibition({ type: 'setInteractionActive', active: !!active });
  }

  function focusIframeWindow() {
    if (!exhibitionFrame || !exhibitionFrame.contentWindow) return;
    try {
      exhibitionFrame.contentWindow.focus();
    } catch (_) {}
  }

  if (exhibitionWrap) {
    exhibitionWrap.addEventListener('mouseenter', function () {
      focusIframeWindow();
      setIframeInteractionActive(true);
    });
    exhibitionWrap.addEventListener('mouseleave', function () {
      setIframeInteractionActive(false);
      try {
        window.focus();
      } catch (_) {}
    });
  }

  exhibitionFrame.addEventListener('load', function () {
    setIframeInteractionActive(false);
  });

  function bindImportCards(scaleSlider) {
    document.querySelectorAll('.btn-import').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var glbPath = this.getAttribute('data-glb');
        if (!glbPath) return;
        fetch(glbPath)
          .then(function (r) {
            return r.ok ? r.arrayBuffer() : Promise.reject(new Error('fetch failed'));
          })
          .then(function (arrayBuffer) {
            var fileName = glbPath.split('/').pop() || 'model.glb';
            var scaleValue = scaleSlider ? parseFloat(scaleSlider.value) || 100 : 100;
            postToExhibition({
              type: 'loadGLB',
              arrayBuffer: arrayBuffer,
              fileName: fileName,
              scale: scaleValue
            });
          })
          .catch(function (err) {
            console.warn('Quick import failed:', glbPath, err);
          });
      });
    });
  }

  var selectFileBtn = document.getElementById('selectFileBtn');
  var fileInput = document.getElementById('glb-file');
  var fileNameEl = document.getElementById('fileName');
  var scaleSlider = document.getElementById('scale-slider');
  var pedestalSlider = document.getElementById('pedestal-slider');

  bindImportCards(scaleSlider);

  if (
    !selectFileBtn ||
    !fileInput ||
    !fileNameEl ||
    !scaleSlider ||
    !pedestalSlider
  ) {
    return;
  }

  selectFileBtn.addEventListener('click', function () {
    fileInput.click();
  });

  fileInput.addEventListener('change', function (e) {
    var f = e.target.files && e.target.files[0];
    if (!f || !f.name.toLowerCase().endsWith('.glb')) return;
    var r = new FileReader();
    r.onload = function () {
      postToExhibition({
        type: 'loadGLB',
        arrayBuffer: r.result,
        fileName: f.name,
        scale: parseFloat(scaleSlider.value) || 100
      });
    };
    r.readAsArrayBuffer(f);
    fileNameEl.textContent = f.name;
    e.target.value = '';
  });

  function syncScale() {
    var v = parseInt(scaleSlider.value, 10);
    v = Math.max(1, Math.min(200, v));
    scaleSlider.value = v;
    postToExhibition({ type: 'setScale', value: v });
  }

  function syncPedestal() {
    var v = parseInt(pedestalSlider.value, 10);
    v = Math.max(5, Math.min(100, v));
    pedestalSlider.value = v;
    postToExhibition({ type: 'setPedestal', value: v });
  }

  var scaleReset = document.getElementById('scale-reset');
  var pedestalReset = document.getElementById('pedestal-reset');

  scaleSlider.addEventListener('input', function () {
    syncScale();
  });

  pedestalSlider.addEventListener('input', function () {
    syncPedestal();
  });

  function bindRotateButton(id, direction) {
    var el = document.getElementById(id);
    if (!el) return;
    function start() {
      postToExhibition({ type: 'setRotationSpeed', value: DEFAULT_ROTATION_SPEED });
      postToExhibition({ type: 'rotate', direction: direction, active: true });
    }
    function end() {
      postToExhibition({ type: 'rotate', direction: direction, active: false });
    }
    el.addEventListener('mousedown', start);
    el.addEventListener('mouseup', end);
    el.addEventListener('mouseleave', end);
    el.addEventListener('touchstart', function (event) {
      event.preventDefault();
      start();
    });
    el.addEventListener('touchend', function (event) {
      event.preventDefault();
      end();
    });
    el.addEventListener('touchcancel', function (event) {
      event.preventDefault();
      end();
    });
  }

  bindRotateButton('rot-left', 'left');
  bindRotateButton('rot-right', 'right');
  bindRotateButton('rot-up', 'up');
  bindRotateButton('rot-down', 'down');

  if (scaleReset) {
    scaleReset.addEventListener('click', function () {
      scaleSlider.value = 100;
      syncScale();
    });
  }
  if (pedestalReset) {
    pedestalReset.addEventListener('click', function () {
      pedestalSlider.value = 32;
      syncPedestal();
    });
  }
})();
