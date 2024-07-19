
-- @generic T
-- @params object: { values: T[] }
function get(object)
  return object.values
end

local obj = {
  values = {1, 2, 3},
}

local xs = get(obj)
return xs -- integer[]

