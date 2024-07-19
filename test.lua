
local x = {
  deep = {
    value = 5,
  }
}
x.deep.value = "" -- type mismatch now flagged by the typechecker
