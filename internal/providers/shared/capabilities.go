package providershared

func NewAPISupportSet(apiTypes []string) map[string]struct{} {
	if len(apiTypes) == 0 {
		return nil
	}
	out := make(map[string]struct{}, len(apiTypes))
	for _, apiType := range apiTypes {
		if apiType == "" {
			continue
		}
		out[apiType] = struct{}{}
	}
	return out
}

func SupportsAPIType(supported map[string]struct{}, apiType string) bool {
	if len(supported) == 0 {
		return true
	}
	_, ok := supported[apiType]
	return ok
}
