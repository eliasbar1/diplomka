const express = require('express')
const app = express()
const port = 3001

app.use(express.static('data'))
app.use('/data', express.static(__dirname + '/data'));

app.get('/', (req, res) => res.sendFile(__dirname + "/index.html"))

app.listen(port, () => console.log(`Cryptomania app listening on port ${port}!`))
