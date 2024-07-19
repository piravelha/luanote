
local obj = {
  deep = {
    deeper = {
      x = 5,
      s = "hello",
    },
  },
}

obj.deep.deeper.x = "invalid"
obj.deep.deeper.s = "world"
