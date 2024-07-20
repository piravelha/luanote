
function add(x)
    --@param x: boolean
    return x and 1
end

--@type boolean | nil | string
local booly = true

--@type boolean
local bool = true

if type(booly) == "boolean" then
    return add(booly)
end
