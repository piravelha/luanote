
-- @generic T : string | number
-- @params x : T
function f(x)
    return x
end

local x = f(1)
local y = f("hello")
local z = f(true) -- Expect error