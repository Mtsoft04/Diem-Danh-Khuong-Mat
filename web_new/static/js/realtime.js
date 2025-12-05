function updateRealtime() {
    fetch('/api/realtime')
        .then(r => r.json())
        .then(data => {
            const tbody = document.querySelector('#realtimeTable tbody');
            tbody.innerHTML = '';
            data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${row.student_id}</td><td>${row.name}</td><td>${row.time_in}</td>`;
                tbody.appendChild(tr);
            });
        });
    setTimeout(updateRealtime, 2000);
}
updateRealtime();