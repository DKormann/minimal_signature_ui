




document.body.innerHTML = "<h1>minimal signature  some tsutttsdkjsdhfkdsjh</h1>"


fetch("/api/message").then((response) => {
  if (response.ok) {
    response.text().then((text) => {
      document.body.innerHTML += `<p>${text}</p>`;
    });
  } else {
    document.body.innerHTML += `<p>Error: ${response.status}</p>`;
  }
});


// fetch("/api/available").then((response) => {


export {}