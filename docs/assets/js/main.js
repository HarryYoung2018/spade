const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector(".nav-links");
const tocToggle = document.querySelector(".toc-toggle");
const tocPanel = document.querySelector(".toc-panel");

function setOpen(button, panel, open) {
  panel.classList.toggle("open", open);
  button.setAttribute("aria-expanded", String(open));
}

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const open = !navLinks.classList.contains("open");
    setOpen(navToggle, navLinks, open);
    if (open && tocToggle && tocPanel) {
      setOpen(tocToggle, tocPanel, false);
    }
  });

  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => setOpen(navToggle, navLinks, false));
  });
}

if (tocToggle && tocPanel) {
  tocToggle.addEventListener("click", () => {
    const open = !tocPanel.classList.contains("open");
    setOpen(tocToggle, tocPanel, open);
    if (open && navToggle && navLinks) {
      setOpen(navToggle, navLinks, false);
    }
  });

  tocPanel.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => setOpen(tocToggle, tocPanel, false));
  });
}
