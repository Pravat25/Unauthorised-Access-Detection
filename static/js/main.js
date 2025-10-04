// small client enhancements if needed
document.addEventListener('DOMContentLoaded', function(){
  // auto dismiss flash messages after 4s
  setTimeout(()=> {
    const flashes = document.querySelectorAll('.flash');
    flashes.forEach(f => f.style.display='none');
  }, 4000);
});
